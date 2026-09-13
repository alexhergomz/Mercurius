"""Stage C — the RoPE dial: convert rotary dimensions to NoPE, continuously.

Qwen3.5 uses partial RoPE: 64 of head_dim 256 are rotary, built from 32
frequencies duplicated into the rotate_half layout. Dialling down means making
selected frequencies' rotation the identity:

    cos -> 1,  sin -> 0

which is exactly NoPE on those dimensions, with no weight change. Fully
reversible, so a sweep costs nothing but forward passes.

Text-only note: when position_ids is None the model builds a plain arange and
expands it to every mRoPE section identically, so M-RoPE degenerates to standard
1D RoPE and the [11,11,10] split is irrelevant here. It becomes relevant again
only with image tokens present.

Frequency ordering: inv_freq[i] = base^(-2i/dim), so i=0 is the fastest
rotation (most local) and i=31 the slowest (most global). Which end to keep is
an empirical question -- hence the policies.
"""
import torch


def _keep_mask(n_freq, keep, policy):
    """Boolean mask over the n_freq frequencies: True = keep rotating."""
    m = torch.zeros(n_freq, dtype=torch.bool)
    if keep <= 0:
        return m
    if keep >= n_freq:
        return ~m
    if policy == "local":        # keep fastest-rotating (low index)
        m[:keep] = True
    elif policy == "global":     # keep slowest-rotating (high index)
        m[-keep:] = True
    elif policy == "stride":     # keep evenly spaced across the ladder
        idx = torch.linspace(0, n_freq - 1, keep).round().long()
        m[idx] = True
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return m


def _mrope_section(model):
    """[T, H, W] frequency counts, wherever this transformers version hides them."""
    from mercurius.surgery.norm_fusion import get_trunk
    cands = [getattr(model, "config", None), getattr(get_trunk(model), "config", None)]
    cands += [getattr(c, "text_config", None) for c in cands if c is not None]
    for cfg in cands:
        rp = getattr(cfg, "rope_parameters", None)
        if isinstance(rp, dict) and rp.get("mrope_section"):
            return list(rp["mrope_section"])
    return None


def _axis_of_freq(n_freq, mrope_section):
    """Which mRoPE axis (0=T, 1=H, 2=W) each frequency slot belongs to.

    Mirrors Qwen3_5MultiModalRotaryEmbedding.recomposition_frequencies exactly:
    it starts from the T frequencies everywhere, then overwrites the strided
    slices slice(1, section[1]*3, 3) with H and slice(2, section[2]*3, 3) with W.
    For [11, 11, 10] over 32 frequencies this is a clean stride-3 interleave and
    the three sets partition all 32 slots.
    """
    ax = torch.zeros(n_freq, dtype=torch.long)          # default: T
    for dim in (1, 2):
        ax[dim:mrope_section[dim] * 3:3] = dim
    return ax


def install_rope_dial(model, keep_freqs, policy="global", image_keep=None):
    """Identity-ize all but `keep_freqs` rotary frequencies. Returns a restore fn.

    image_keep=None (default) applies the same mask to every token -- this is the
    behaviour every measurement in this project was taken under, so it is left
    bit-identical.

    image_keep="hw" additionally keeps the H and W frequency slots rotating AT
    IMAGE-TOKEN POSITIONS ONLY, while text positions keep the base mask.

    WHY THE GATE MUST BE PER-TOKEN, NOT PER-SECTION. The obvious idea -- "keep
    the H/W sections, zero T" -- does not work, and the reason is in
    get_rope_index: for TEXT the position ids are
        torch.arange(text_len).view(1, -1).expand(3, -1) + current_pos
    i.e. all three axes carry the identical 1D index. So on text the H and W
    slots encode ordinary sequence position, indistinguishable from T, and a
    section mask that keeps H+W leaves 21 of 32 frequencies rotating on text --
    silently undoing most of the NoPE surgery on the language side. There is no
    frequency mask that is NoPE-on-text and RoPE-on-images. Only a per-token gate
    separates them.

    Image tokens are identified from position_ids alone: text has all three axes
    equal, image patches have H/W varying. That needs no extra plumbing (this
    hook never sees input_ids). It misclassifies exactly the first token of each
    image, where t == h == w, which then simply receives the text mask.
    """
    from mercurius.surgery.norm_fusion import get_trunk
    rot = get_trunk(model).rotary_emb
    n_freq = rot.inv_freq.shape[0]
    mask = _keep_mask(n_freq, keep_freqs, policy)
    # cos/sin come back as cat([f, f]) over the last dim
    keep = torch.cat([mask, mask]).to(rot.inv_freq.device)

    img_keep = None
    if image_keep is not None:
        sec = _mrope_section(model)
        if sec is None:
            raise ValueError("image_keep requested but no mrope_section in config")
        if sum(sec) != n_freq:
            raise ValueError(f"mrope_section {sec} sums to {sum(sec)}, "
                             f"but there are {n_freq} frequencies")
        ax = _axis_of_freq(n_freq, sec)
        if image_keep == "hw":
            m_img = ax != 0                      # keep H and W, drop T
        elif image_keep == "all":
            m_img = torch.ones(n_freq, dtype=torch.bool)
        else:
            raise ValueError(f"unknown image_keep {image_keep!r}")
        img_keep = torch.cat([m_img, m_img]).to(rot.inv_freq.device)

    orig_forward = rot.forward

    def dialled(x, position_ids):
        cos, sin = orig_forward(x, position_ids)
        k = keep.to(cos.dtype)
        if img_keep is None:
            return cos * k + (1.0 - k), sin * k
        # position_ids is (3, bs, seq); image tokens have H/W differing from T
        pid = position_ids
        if pid.dim() != 3 or pid.shape[0] != 3:
            # text-only path built a plain arange -- nothing is an image token
            return cos * k + (1.0 - k), sin * k
        is_img = ((pid[1] != pid[0]) | (pid[2] != pid[0])).to(cos.dtype)  # (bs, seq)
        ki = img_keep.to(cos.dtype)
        k_tok = k + is_img[..., None] * (ki - k)          # (bs, seq, 2*n_freq)
        return cos * k_tok + (1.0 - k_tok), sin * k_tok

    rot.forward = dialled
    rot._dial = {"keep_freqs": int(mask.sum()), "policy": policy,
                 "n_freq": n_freq, "rotary_dims": int(mask.sum()) * 2,
                 "image_keep": image_keep}

    def restore():
        rot.forward = orig_forward
        if hasattr(rot, "_dial"):
            del rot._dial
    return restore


def dial_summary(model):
    from mercurius.surgery.norm_fusion import get_trunk
    rot = get_trunk(model).rotary_emb
    d = getattr(rot, "_dial", None)
    if d is None:
        return f"full RoPE ({rot.inv_freq.shape[0]} freqs / "\
               f"{rot.inv_freq.shape[0]*2} rotary dims)"
    return (f"{d['keep_freqs']}/{d['n_freq']} freqs kept "
            f"({d['rotary_dims']}/{d['n_freq']*2} rotary dims, {d['policy']})")
