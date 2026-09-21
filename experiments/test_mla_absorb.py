"""Is what we train and evaluate the same function as what would ship?

Builds the student exactly as the trainer does up to the optimizer (stage A+B,
NoPE, GDN-2, CARE-whitened MLA, per-head query maps), moves the new
parameters OFF their initialisation so the check is not trivially an identity,
and compares:

  1. PerHeadQ against a per-head reference built from the stock
     [q_h | gate_h] layout: every head's query mapped, every gate untouched.
  2. The unabsorbed latent attention (what training and every eval use)
     against the absorbed form (mla_absorb.py), whole-model logits and
     perplexity, in fp32 and in bf16, against a matched control: the bf16
     rounding of the unabsorbed path itself. "Unbiased" means the
     absorbed-vs-unabsorbed gap is at or below that control.

    python experiments/test_mla_absorb.py [--dc 512]
"""
import argparse
import torch
from transformers import AutoTokenizer

from mercurius.models.kda import load_kda_model
from mercurius.models.gdn2 import convert_to_gdn2
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.surgery.transmla import convert_to_mla
from mercurius.surgery.perhead_q import install_per_head_q, PerHeadQ
from mercurius.surgery.mla_absorb import install_absorbed
from mercurius.paths import CACHE_DIR, STAGE_AB, WIKITEXT


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dc", type=int, default=512)
    ap.add_argument("--covs", default=str(CACHE_DIR / "kv_covs_4b.pt"))
    ap.add_argument("--n", type=int, default=2048)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    ok = []

    m = load_kda_model(str(STAGE_AB), dtype=torch.float32)
    install_rope_dial(m, 0, "global")
    convert_to_gdn2(m, verbose=False)
    covs = {int(k): v.cuda().float() for k, v in torch.load(a.covs).items()}
    convert_to_mla(m, d_c=a.dc, covs=covs, verbose=False)
    install_per_head_q(m, verbose=False)
    cfg = getattr(m.config, "text_config", m.config)
    H, G, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    for l in get_trunk(m).layers:
        sa = getattr(l, "self_attn", None)
        if sa is None:
            continue
        sa.q_proj.R.add_(0.05 * torch.randn_like(sa.q_proj.R))
        lat = sa.k_proj.latent
        lat.up_k.weight.mul_(1 + 0.05 * torch.randn_like(lat.up_k.weight))
        sa.k_norm.weight.add_(0.05 * torch.randn_like(sa.k_norm.weight))

    print("1. per-head query layout")
    sa = next(l.self_attn for l in get_trunk(m).layers if hasattr(l, "self_attn"))
    x = torch.randn(1, 7, cfg.hidden_size, device="cuda")
    got = sa.q_proj(x).view(1, 7, H, 2 * d)
    raw = sa.q_proj.base(x).view(1, 7, H, 2 * d)
    want_q = torch.einsum("bthd,hde->bthe", raw[..., :d], sa.q_proj.R)
    e_q, e_g = rel(got[..., :d], want_q), rel(got[..., d:], raw[..., d:])
    ok.append(e_q < 1e-6 and e_g == 0.0)
    print(f"  queries vs per-head reference relL2 {e_q:.1e}; gates changed "
          f"{e_g:.1e}  [{'PASS' if ok[-1] else 'FAIL'}]")
    moved = (got[..., :d] - raw[..., :d]).flatten(0, 1).norm(dim=-1).gt(0).all().item()
    ok.append(moved)
    print(f"  every head's query is mapped: {moved}  [{'PASS' if moved else 'FAIL'}]")

    print("2. absorbed vs unabsorbed")
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    ids = tok(open(WIKITEXT).read()[:40000], return_tensors="pt").input_ids[0][:a.n]
    x = ids.unsqueeze(0).cuda()

    def run():
        lg = m(input_ids=x, use_cache=False).logits[0].float()
        ce = torch.nn.functional.cross_entropy(lg[:-1], x[0, 1:])
        return lg, ce.exp().item()

    ref, p_ref = run()
    # Control: the unabsorbed path with every MLA weight (down, up_k, up_v,
    # k_norm, R) perturbed by one fp32 ULP, relative 2^-23. Absorption only
    # reassociates products of exactly these tensors, so "unbiased" means its
    # error is at this scale. fla's kernels run TF32 internally whatever
    # torch's flags say and the model amplifies over 32 layers, so an
    # absolute threshold would be a guess.
    saved = []
    for l in get_trunk(m).layers:
        sa = getattr(l, "self_attn", None)
        if sa is None:
            continue
        lat = sa.k_proj.latent
        for p_ in (lat.down.weight, lat.up_k.weight, lat.up_v.weight,
                   sa.k_norm.weight, sa.q_proj.R):
            saved.append((p_, p_.data.clone()))
            p_.data.mul_(1 + 2.0 ** -23 * torch.randn_like(p_))
    alt, _ = run()
    for p_, v in saved:
        p_.data.copy_(v)
    c32 = rel(alt, ref)
    restore = install_absorbed(m, torch.float32)
    ab, p_ab = run()
    restore()
    e32 = rel(ab, ref)
    ok.append(e32 <= 1.5 * c32)
    print(f"  fp32: absorbed vs unabsorbed relL2 {e32:.2e}; control (1-ULP "
          f"MLA weight noise) {c32:.2e}; ppl {p_ref:.5f} vs {p_ab:.5f}  "
          f"[{'PASS' if ok[-1] else 'FAIL'}]")

    m.to(torch.bfloat16)
    for mod in m.modules():                   # R stays a trainable fp32 param
        if isinstance(mod, PerHeadQ):
            mod.R.data = mod.R.data.float()
    ref16, p16 = run()
    restore = install_absorbed(m, torch.bfloat16)
    ab16, p_ab16 = run()
    restore()
    ctrl, gap = rel(ref16, ref), rel(ab16, ref16)
    ok.append(gap <= 1.5 * ctrl)
    print(f"  bf16: absorbed vs unabsorbed relL2 {gap:.2e}; control (bf16 "
          f"rounding of the unabsorbed path itself) {ctrl:.2e}; ppl "
          f"{p16:.4f} vs {p_ab16:.4f}  [{'PASS' if ok[-1] else 'FAIL'}]")

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
