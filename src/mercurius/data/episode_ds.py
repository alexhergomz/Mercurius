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

import torch


class EpisodeDataset:
    def __init__(self, path, tok, cache_dir=None, verbose=True):
        self.path = path
        key = hashlib.sha1(
            f"{os.path.abspath(path)}:{os.path.getmtime(path)}:"
            f"{getattr(tok, 'name_or_path', '')}".encode()).hexdigest()[:16]
        cache = os.path.join(cache_dir or os.path.dirname(path), f".ids-{key}.pt")
        if os.path.exists(cache):
            blob = torch.load(cache)
            self.ids, self.meta = blob["ids"], blob["meta"]
        else:
            self.ids, self.meta = [], []
            with open(path) as fh:
                for line in fh:
                    ep = json.loads(line)
                    text = tok.apply_chat_template(ep["messages"], tools=ep.get("tools"),
                                                   tokenize=False)
                    ids = tok(text, add_special_tokens=False).input_ids
                    self.ids.append(torch.tensor(ids, dtype=torch.long))
                    self.meta.append({"repo": ep.get("repo"), "license": ep.get("license"),
                                      "n_tokens": len(ids)})
            torch.save({"ids": self.ids, "meta": self.meta}, cache)
        self.lengths = torch.tensor([len(x) for x in self.ids])
        if verbose and len(self.ids):
            q = self.lengths.float().quantile(torch.tensor([0.25, 0.5, 0.75])).tolist()
            print(f"  episodes: {len(self.ids):,} from {os.path.basename(path)}, "
                  f"{int(self.lengths.sum()):,} tokens (p25/median/p75 "
                  f"{int(q[0]):,}/{int(q[1]):,}/{int(q[2]):,}, max "
                  f"{int(self.lengths.max()):,})", flush=True)

    def __len__(self):
        return len(self.ids)

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
