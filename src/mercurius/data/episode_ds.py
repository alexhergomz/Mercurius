"""Agent episodes as training sequences, rendered exactly as the model sees them.

Each episode (data/episodes/*.jsonl) is rendered through the student's chat
template WITH ITS TOOL SCHEMAS, so a training sequence is byte-for-byte the
deployed format: tool definitions in the system prompt, <think> blocks,
<tool_call> XML, <tool_response> results.

One episode per step, at its natural length -- the trainer runs one sequence
per step, so a 6k-token episode simply trains on 6k tokens. No padding (which
would waste prefill on nothing) and no packing (which would need variable-length
kernels and a state reset at every boundary; worth doing only if the waste
turns out to matter). An episode longer than the window keeps its FIRST `seq`
tokens: that keeps the system prompt with the tool schemas, the task, and the
early turns, which is the part that makes the sequence an agent episode at all.

Rendered ids are cached next to the jsonl, keyed by the tokenizer's name and
the file's mtime, because rendering 10k episodes is minutes of CPU.
"""
import hashlib
import json
import os
import re

import torch


# The closing <|im_end|> is INSIDE the captured group on purpose. With it
# outside, the assistant's content was supervised but the token that ENDS the
# turn never was -- so the model got no gradient for stopping. Combined with a
# raw-text majority that contains no end token at all (measured: 0 in 1.38M
# tokens of fineweb_edu_long), the recovery run supervised termination exactly
# zero times, and the base model's instruction-tuned stopping behaviour decayed.
# Measured 2026-09-24: base model 0% non-terminating generations, recovered
# model 14.6% -- every failure running to whatever budget it was given (7 of 48
# at both a 2048 and a 6144 cap, against a 637-token maximum among healthy ones).
# bump when assistant_mask changes what it marks; see the cache key below
MASK_VERSION = 2

ASSISTANT_SPAN = re.compile(r"<\|im_start\|>assistant\n(.*?(?:<\|im_end\|>|\Z))", re.S)


def assistant_mask(ep, tok, n_tokens, text=None):
    """True on ASSISTANT tokens only; False on tool results, system and user.

    Tool results are PREFILL CONTEXT, not targets. The environment inserts them;
    the model never generates them, so supervising their prediction supervises a
    behaviour that does not occur. Measured on our corpus: 84.7% of episode
    tokens are tool output and only 8.4% are assistant turns, so an unmasked loss
    spends twelve times more gradient teaching the model to reproduce file
    contents than to decide what to do -- and the measured failure mode on
    held-out tasks was precisely that it explores and then fails to emit an
    answer, which lives in those 8.4%.

    Spans are found by CHARACTER OFFSET in the fully rendered text and mapped to
    tokens through the tokenizer's offset mapping. Rendering message prefixes
    incrementally does not work: the chat template refuses a prefix with no user
    turn, and it inserts its own control tokens, so the boundaries would not line
    up.
    """
    if text is None:
        text = tok.apply_chat_template(ep["messages"], tools=ep.get("tools"),
                                       tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    mask = torch.zeros(min(n_tokens, len(offs)), dtype=torch.bool)
    spans = [(m.start(1), m.end(1)) for m in ASSISTANT_SPAN.finditer(text)]
    for i, (a, b) in enumerate(offs):
        if i >= len(mask):
            break
        if b > a and any(a >= s0 and b <= s1 for s0, s1 in spans):
            mask[i] = True
    if len(mask) < n_tokens:
        mask = torch.cat([mask, torch.zeros(n_tokens - len(mask), dtype=torch.bool)])
    return mask


class EpisodeDataset:
    def __init__(self, path, tok, cache_dir=None, verbose=True):
        self.path = path
        # MASK_VERSION is part of the key because the cache stores the MASKS,
        # not just the token ids. Without it, changing how a mask is computed
        # leaves every existing cache valid and the change silently does
        # nothing -- which is how the terminator fix below would have been
        # nullified on a machine that had already built the cache.
        key = hashlib.sha1(
            f"{os.path.abspath(path)}:{os.path.getmtime(path)}:"
            f"{getattr(tok, 'name_or_path', '')}:{MASK_VERSION}".encode()).hexdigest()[:16]
        cache = os.path.join(cache_dir or os.path.dirname(path), f".ids-{key}.pt")
        if os.path.exists(cache):
            blob = torch.load(cache)
            self.ids, self.meta = blob["ids"], blob["meta"]
            self.masks = blob.get("masks") or [torch.ones_like(x, dtype=torch.bool)
                                               for x in self.ids]
        else:
            self.ids, self.meta, self.masks = [], [], []
            with open(path) as fh:
                for line in fh:
                    ep = json.loads(line)
                    text = tok.apply_chat_template(ep["messages"], tools=ep.get("tools"),
                                                   tokenize=False)
                    ids = tok(text, add_special_tokens=False).input_ids
                    self.ids.append(torch.tensor(ids, dtype=torch.long))
                    self.masks.append(assistant_mask(ep, tok, len(ids)))
                    self.meta.append({"repo": ep.get("repo"), "license": ep.get("license"),
                                      "n_tokens": len(ids)})
            torch.save({"ids": self.ids, "meta": self.meta, "masks": self.masks}, cache)
        self.lengths = torch.tensor([len(x) for x in self.ids])
        if verbose and len(self.ids):
            q = self.lengths.float().quantile(torch.tensor([0.25, 0.5, 0.75])).tolist()
            print(f"  episodes: {len(self.ids):,} from {os.path.basename(path)}, "
                  f"{int(self.lengths.sum()):,} tokens (p25/median/p75 "
                  f"{int(q[0]):,}/{int(q[1]):,}/{int(q[2]):,}, max "
                  f"{int(self.lengths.max()):,})", flush=True)

    def __len__(self):
        return len(self.ids)

    def sample_with_mask(self, gen, seq, min_len=512, tries=8):
        """(ids, mask) for one episode, windowed so the window CONTAINS
        supervised tokens.

        Keeping the first `seq` tokens is right for an agent episode -- system
        prompt, task, early turns -- and wrong for a commit-diff task, where the
        prompt is a whole source file and the TARGET is the diff at the end. At
        seq 2048 every diff task truncated to prompt-only: the mask came back
        entirely False, the loss was exactly 0.0, and perplexity overflowed.

        So: if the head window has no supervised token, slide the window to end
        at the last supervised one. If even that fails, draw another episode.
        """
        if not self.ids:
            return None, None
        for _ in range(tries):
            w = self.lengths.clamp(max=seq).float()
            i = int(torch.multinomial(w, 1, generator=gen))
            x, m = self.ids[i], self.masks[i]
            if len(x) < min_len:
                continue
            if len(x) <= seq:
                if bool(m.any()):
                    return x, m
                continue
            if bool(m[:seq].any()):
                return x[:seq], m[:seq]
            idx = torch.nonzero(m, as_tuple=False)
            if idx.numel() == 0:
                continue
            end = int(idx[-1]) + 1
            start = max(0, end - seq)
            xs, ms = x[start:start + seq], m[start:start + seq]
            if bool(ms.any()) and len(xs) >= min_len:
                return xs, ms
        return None, None

    def sample(self, gen, seq, min_len=512):
        """One episode, truncated to `seq`. Sampling is by TOKEN COUNT, so a
        long episode is not under-represented relative to the corpus windows it
        competes with."""
        if not self.ids:
            return None
        w = self.lengths.clamp(max=seq).float()
        i = int(torch.multinomial(w, 1, generator=gen))
        x = self.ids[i]
        if len(x) < min_len:
            return None
        return x[:seq]
