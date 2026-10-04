"""Give the student the TEACHER's output head, behind a small projection.

    logits = W_t (P h_s + b)        P: d_s -> d_t trainable, W_t frozen

WHY. Distilling the final hidden state is only equivalent to distilling the
distribution when both models decode through the SAME head -- h -> W h is
injective for V >> d, so matching h and matching logits are the same constraint
in different coordinates, but only in a shared basis. Mapping the teacher back
into the STUDENT's head instead was measured and rejected: on 321 real teacher
states it loses 18% of top-1 predictions (KL 0.433) where 5% random noise on h
costs KL 0.0017. Decoding through the teacher's own head has zero reconstruction
error by construction.

The teacher (35B-A3B) is NARROWER than the student, 2048 vs 2560, so this is
also cheap: the replacement head is 508.6M against the student's own 635.7M, plus
5.2M for P. Keeping it FACTORED matters -- fusing W_t P into one (V, d_s) matrix
is equally exact but stores a rank-2048 map densely, costing 635.7M instead of
513.8M.

INIT. P is fitted to preserve the STUDENT's own logits (W_t P ~= W_s), usage-
weighted over the training vocabulary, NOT to match h_t. Measured, held out:

    fit P h_s ~= h_t                  disruption KL 0.7401  top-1 74.0%
    fit W_t P ~= W_s, uniform tokens             KL 1.0461  top-1 66.3%
    fit W_t P ~= W_s, usage-weighted             KL 0.0663  top-1 94.2%

Fitting P to the training target destroys the model it starts from -- it costs 14
points of teacher agreement and nearly doubles output entropy. The rank argument
(2048 < 2560 cannot be exact) does NOT make this lossy in practice; UNIFORM token
weighting does, catastrophically. Built by scripts/make_head_swap.py.
"""
import torch
import torch.nn as nn


class ProjectedHead(nn.Module):
    """Trainable d_s -> d_t projection composed with the teacher's frozen head."""

    def __init__(self, P, b, W_t, dtype=torch.bfloat16):
        super().__init__()
        d_t, d_s = P.shape
        V = W_t.shape[0]
        self.d_s, self.d_t, self.V = d_s, d_t, V
        self.proj = nn.Linear(d_s, d_t, bias=True, dtype=dtype)
        with torch.no_grad():
            self.proj.weight.copy_(P.to(dtype))
            self.proj.bias.copy_(b.to(dtype))
        self.head = nn.Linear(d_t, V, bias=False, dtype=dtype)
        with torch.no_grad():
            self.head.weight.copy_(W_t.to(dtype))
        self.head.weight.requires_grad_(False)          # the teacher's, frozen

    def forward(self, h):
        return self.head(self.proj(h.to(self.proj.weight.dtype)))

    def project(self, h):
        """h_s -> teacher space. This is what the hidden-state loss compares."""
        return self.proj(h.to(self.proj.weight.dtype))

    @property
    def weight(self):
        raise AttributeError(
            "ProjectedHead has no single weight matrix: logits = W_t(P h + b). "
            "Use .head.weight for the teacher's head, .proj for the projection, "
            "or .project(h) for the hidden-state target space.")


def apply_head_swap(model, blob, dtype=torch.bfloat16, train_proj=True):
    """Replace the student's tied output head with the teacher's, behind P.

    Returns the ProjectedHead. The input embedding is untouched: the student
    still embeds with its own matrix, only the OUTPUT side moves to the
    teacher's basis, so the model is untied by this operation.
    """
    P, W_t = blob["P"], blob["W_t"]
    b = blob.get("b")
    if b is None:
        b = torch.zeros(P.shape[0])
    head = ProjectedHead(P, b, W_t, dtype=dtype)
    dev = next(model.parameters()).device
    head = head.to(dev)
    if hasattr(model, "lm_head"):
        model.lm_head = head
    else:
        model.set_output_embeddings(head)
    if hasattr(model, "config"):
        model.config.tie_word_embeddings = False
        inner = getattr(model.config, "text_config", None)
        if inner is not None:
            inner.tie_word_embeddings = False
    for p in head.head.parameters():
        p.requires_grad_(False)
    for p in head.proj.parameters():
        p.requires_grad_(bool(train_proj))
    n_tr = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"  head swap: logits = W_t{tuple(head.head.weight.shape)} "
          f"@ P{tuple(head.proj.weight.shape)}; {n_tr:,} trainable, "
          f"{head.head.weight.numel():,} frozen (teacher's)", flush=True)
    return head
