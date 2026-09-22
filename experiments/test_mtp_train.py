"""The conv MTP head must train stably on the student's real hidden states.

Trains the head ALONE (trunk frozen, hidden states precomputed) for N steps at
the trainer's MTP learning rate, with the trainer's optimizer (bnb AdamW8bit)
and gradient clipping, on CE against the true future tokens. Passes if the
loss never spikes (no step more than 1.5x the previous) and ends below where
it started, and every head ends better than the identity init -- which is
simply the t+1 prediction reused for t+1+j.

    python experiments/test_mtp_train.py
"""
import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from transformers import AutoTokenizer
from mercurius import guard
from mercurius.eval.retrieval_ab import build
from mercurius.models.mtp_conv import ConvMTPHead
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.paths import CKPT_DIR, CACHE_DIR, STAGE_AB, FINEWEB_LONG

guard.cap_cuda_memory(40)
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
m = build(str(CKPT_DIR / "adapters-4b27b-s8192-best.pt"), 512, str(CACHE_DIR / "kv_covs_4b.pt"), quantize=True)
ids = tok(open(FINEWEB_LONG).read(2_000_000), return_tensors="pt").input_ids[0]
T, NW, K, STEPS = 2048, 12, 4, 60
wins = [ids[i * T:(i + 1) * T].cuda() for i in range(NW)]
with torch.no_grad():
    H = [get_trunk(m)(input_ids=w.unsqueeze(0)).last_hidden_state for w in wins]
W = m.get_output_embeddings().weight
head = ConvMTPHead(d_model=H[0].shape[-1], k=K)
with torch.no_grad():                    # internal scale must stay bounded
    xx = H[0].transpose(1, 2)
    scales = [xx.float().abs().max().item()]
    for blk in head.blocks:
        xx = blk(xx)
        scales.append(xx.float().abs().max().item())
    z0 = head(H[0])
print(f"  max|x| through the blocks: {' -> '.join(f'{v:.1f}' for v in scales)}; "
      f"identity at init max|z-h| {(z0 - H[0].unsqueeze(2)).abs().max().item():.1e}")
opt = bnb.optim.AdamW8bit(head.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)


def loss_per_head(h, x):
    z = head(h)[0]                                   # (T, K, d)
    out = []
    for j in range(1, K + 1):
        lg = (z[:T - 1 - j, j - 1] @ W.T).float()
        out.append(F.cross_entropy(lg, x[1 + j:]))
    return torch.stack(out)


with torch.no_grad():
    init = torch.stack([loss_per_head(H[i], wins[i]) for i in range(NW - 2, NW)]).mean(0)
hist = []
for step in range(STEPS):
    i = step % (NW - 2)                                # last 2 windows held out
    l = loss_per_head(H[i], wins[i]).mean()
    opt.zero_grad(); l.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    opt.step()
    hist.append(l.item())
    if step % 10 == 0 or step == STEPS - 1:
        print(f"  step {step:>3} train CE (mean over heads) {l.item():.4f}")
with torch.no_grad():
    fin = torch.stack([loss_per_head(H[i], wins[i]) for i in range(NW - 2, NW)]).mean(0)
spikes = [k for k in range(1, len(hist)) if hist[k] > 1.5 * hist[k - 1]]
ok = not spikes and hist[-1] < hist[0] and bool((fin < init).all())
print(f"held-out CE per head (t+2..t+{K+1}): init {[round(v, 3) for v in init.tolist()]}")
print(f"                                   final {[round(v, 3) for v in fin.tolist()]}")
print(f"spikes (>1.5x previous step): {spikes}  [{'PASS' if ok else 'FAIL'}]")
raise SystemExit(0 if ok else 1)
