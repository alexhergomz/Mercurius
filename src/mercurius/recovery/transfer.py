"""Attention transfer: repair the dialled attention layers layer-locally.

Adapted from LoLCATs (ICLR 2025), which linearizes Llama 3 8B in hours on a
single A100 by training each converted layer to match its original counterpart
BEFORE any end-to-end work.

Our problem is far smaller than theirs. They approximate softmax attention with
linear attention -- hard. We only remove positional encoding from 6 layers;
GDN->KDA is already exact. So only those 6 layers need repair.

Why this is cheap, versus end-to-end KL:

  signal     full hidden vector per token, not one softmax over 248,320 classes
  backprop   through ONE layer, not 24
  student    6 attention layers, not the whole model
  teacher    one forward, reused for all 6 layers simultaneously

The teacher forward gives hidden_states[i] (layer input) and hidden_states[i+1]
(layer output) for free via output_hidden_states=True. We then run only the
student's layer i on the teacher's input and match its output. Errors do not
compound across layers here -- that is what the short end-to-end pass afterwards
is for.
"""
import sys, json, time, argparse, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from mercurius.models.kda import load_kda_model
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.adapters.lora import inject_lora, freeze_base, trainable_parameters
from mercurius.eval.suite import ce_and_topk, sample, report, PROMPTS
import bitsandbytes as bnb
from mercurius.paths import BASE_MODEL, FINEWEB, STAGE_AB, WIKITEXT

CKPT = str(STAGE_AB)
ORIG = str(BASE_MODEL)
TRAIN_DATA = str(FINEWEB)
EVAL_DATA = str(WIKITEXT)

# only the layers we actually damaged
ATTN_LORA = [("self_attn.q_proj", 32), ("self_attn.k_proj", 32),
             ("self_attn.v_proj", 32), ("self_attn.o_proj", 32)]


def attn_layer_indices(model):
    return [i for i, l in enumerate(get_trunk(model).layers)
            if hasattr(l, "self_attn")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dial", default="nope", choices=["nope", "c1", "c0"])
    ap.add_argument("--seed-decay", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--length-mix", action="store_true",
                    help="draw sequence length per step (Dataset Decomposition)")
    ap.add_argument("--lm-weight", type=float, default=0.0,
                    help="add cross-entropy on real text. KL/MSE to the teacher "
                         "CAPS quality at the teacher; KDA's channel-wise decay "
                         "is new capacity meant to EXCEED it, so a teacher-free "
                         "term is needed for that capacity to pay off.")
    ap.add_argument("--tag", default="transfer")
    a = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    train_ids = tok(open(TRAIN_DATA).read(), return_tensors="pt").input_ids[0]
    eval_ids = tok(open(EVAL_DATA).read(), return_tensors="pt").input_ids[0]
    print(f"train {len(train_ids):,} tok | eval {len(eval_ids):,} tok", flush=True)

    teacher = AutoModelForCausalLM.from_pretrained(
        ORIG, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = load_kda_model(CKPT, dtype=torch.bfloat16)
    if a.seed_decay:
        for l in get_trunk(student).layers:
            if hasattr(l, "linear_attn"):
                l.linear_attn.seed_decay_from_rope()
    keep, policy = {"nope": (0, "global"), "c1": (16, "local"),
                    "c0": (32, "local")}[a.dial]
    install_rope_dial(student, keep, policy)

    inject_lora(student, ATTN_LORA)
    n_tr = freeze_base(student)
    params = trainable_parameters(student)
    idxs = attn_layer_indices(student)
    print(f"  repairing layers {idxs}", flush=True)
    print(f"  trainable: {n_tr/1e6:.2f} M params", flush=True)

    opt = bnb.optim.AdamW8bit(params, lr=a.lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)

    s_layers = get_trunk(student).layers
    rot = get_trunk(student).rotary_emb
    hist = {"loss": [], "eval": []}

    def evaluate(step):
        student.eval()
        row = {"step": step}
        print(f"  [eval @ {step:>4}]", flush=True)
        for n in (2048, 8192):
            m = ce_and_topk(student, eval_ids, n, teacher=teacher)
            row[n] = m
            report(f"@{n}", m)
        hist["eval"].append(row)
        student.train()

    evaluate(0)
    g = torch.Generator().manual_seed(0)
    t0, seen = time.perf_counter(), 0
    student.train()
    MIX = [(1024, 0.35), (2048, 0.30), (8192, 0.25), (32768, 0.10)]

    def draw_len():
        if not a.length_mix:
            return a.seq
        r, acc = float(torch.rand(1, generator=g)), 0.0
        for n, p in MIX:
            acc += p
            if r <= acc:
                return n
        return MIX[-1][0]

    for step in range(1, a.steps + 1):
        L = draw_len()
        i = int(torch.randint(0, len(train_ids) - L - 1, (1,), generator=g))
        x = train_ids[i:i + L].unsqueeze(0).cuda()
        seen += x.numel()

        # ONE teacher forward supplies input/target for all 6 layers at once
        with torch.no_grad():
            th = teacher(input_ids=x, output_hidden_states=True).hidden_states

        # position embeddings under the student's dial (identity for NoPE)
        pos = torch.arange(L, device=x.device).view(1, 1, -1).expand(4, 1, -1)
        cos, sin = rot(th[0], pos)

        loss = 0.0
        for li in idxs:
            inp = th[li].detach()                 # teacher's input to layer li
            tgt = th[li + 1].detach()             # teacher's output of layer li
            out = s_layers[li](hidden_states=inp, position_embeddings=(cos, sin),
                               attention_mask=None)
            if isinstance(out, tuple):
                out = out[0]
            loss = loss + F.mse_loss(out.float(), tgt.float())
        loss = loss / len(idxs)

        # Teacher-free term. MSE/KL to the teacher CAPS the student at teacher
        # quality; KDA's channel-wise decay is capacity meant to EXCEED it, so
        # without this the lift can only ever reproduce GDN behaviour.
        if a.lm_weight > 0:
            s_logits = student(input_ids=x).logits[0]
            ce = F.cross_entropy(s_logits[:-1].float(), x[0, 1:])
            loss = loss + a.lm_weight * ce
            del s_logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        hist["loss"].append(loss.item())
        del th
        if step % 25 == 0:
            el = time.perf_counter() - t0
            print(f"  step {step:>4}/{a.steps}  MSE {loss.item():9.5f}  "
                  f"{seen/el:6.1f} tok/s  {el/60:5.1f} min", flush=True)
        if step % a.eval_every == 0:
            evaluate(step)
        torch.cuda.empty_cache()

    evaluate(a.steps)
    out = f"logs/transfer-{a.tag}.json"
    json.dump({"args": vars(a), **hist}, open(out, "w"), indent=2)
    torch.save({k: v.detach().cpu() for k, v in student.state_dict().items()
                if "lora_" in k},
               f"ckpt/adapters-{a.tag}.pt")
    print(f"\nwrote {out}")

    e0, e1 = hist["eval"][0], hist["eval"][-1]
    print("\n=== ATTENTION TRANSFER ===")
    for n in (2048, 8192):
        print(f"  @{n}: CE {e0[n]['ce']:.4f} -> {e1[n]['ce']:.4f} | "
              f"ppl {e0[n]['ppl']:.3f} -> {e1[n]['ppl']:.3f} | "
              f"teacher-agree top1 {e0[n]['agree_top1']:.2f} -> "
              f"{e1[n]['agree_top1']:.2f}%")
    print(f"  tokens seen: {seen/1e6:.2f} M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
