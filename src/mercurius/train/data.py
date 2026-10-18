"""The training mix: windows from a document corpus, chat episodes and math packs, with a per-step length mix.

One uniform draw per step picks the source: [0, ep) an episode, [ep, ep + math) a math pack, else a corpus window.
Corpus windows are drawn INSIDE one document (doc_aware), from documents long enough for the drawn length; the
length mix applies to corpus windows only. Episodes and math are supervised on assistant tokens only (both the
divergence and the data term).

Tokenized corpora are cached under the run's cache directory (keyed by file size / mtime and tokenizer), so a
restart does not re-tokenize tens of millions of tokens. Every random stream is a torch.Generator whose state goes
into the resume file, so a resumed run continues the same draw sequence.
"""
from __future__ import annotations

import hashlib
import os

import torch


def _key(path, tok, extra=""):
    st = os.stat(path)
    return hashlib.sha1(f"{os.path.abspath(path)}:{st.st_size}:{int(st.st_mtime)}:"
                        f"{getattr(tok, 'name_or_path', '')}:{len(tok)}:{extra}".encode()).hexdigest()[:16]


def tokenize_documents(path, tok, cache_dir=None, log=print):
    """-> (stream LongTensor, doc spans [(start, end)]) for a text file of documents separated by blank lines."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cp = os.path.join(cache_dir, f"docs-{os.path.basename(path)}-{_key(path, tok)}.pt")
        if os.path.exists(cp):
            b = torch.load(cp)
            return b["stream"].long(), b["spans"]
    docs = [d for d in open(path, encoding="utf-8", errors="replace").read().split("\n\n") if d.strip()]
    chunks, spans, pos = [], [], 0
    for d in docs:
        ids = tok(d, add_special_tokens=False).input_ids
        chunks.append(torch.tensor(ids, dtype=torch.int32))
        spans.append((pos, pos + len(ids)))
        pos += len(ids)
    stream = torch.cat(chunks)
    log(f"  corpus {os.path.basename(path)}: {len(docs):,} documents, {pos / 1e6:.1f} M tokens")
    if cache_dir:
        torch.save({"stream": stream, "spans": spans}, cp)
    return stream.long(), spans


class TrainingMix:
    def __init__(self, data_cfg, tok, cache_dir=None, seed=0, log=print):
        self.cfg, self.tok = data_cfg, tok
        cache_dir = cache_dir if data_cfg.tokenized_cache else None
        self.stream, spans = tokenize_documents(data_cfg.text, tok, cache_dir, log)
        self.doc_aware = data_cfg.doc_aware
        lens = torch.tensor([e - s for s, e in spans])
        order = torch.argsort(lens)
        self.span_start = torch.tensor([s for s, _ in spans])[order]
        self.span_len = lens[order]                                       # ascending
        self.ep = self._episodes(data_cfg.episodes, cache_dir, log)
        self.math = self._episodes(data_cfg.math, cache_dir, log)
        self.ep_frac = float(data_cfg.episodes["frac"]) if self.ep else 0.0
        self.math_frac = float(data_cfg.math["frac"]) if self.math else 0.0
        self.mix = sorted(data_cfg.length_mix.items()) if data_cfg.length_mix else None
        self.g_len = torch.Generator().manual_seed(1234 + seed)
        self.g_win = torch.Generator().manual_seed(seed)
        self.g_src = torch.Generator().manual_seed(4242 + seed)
        self.counts = {"text": 0, "episodes": 0, "math": 0}
        longest = max([data_cfg.seq] + [n for n, _ in (self.mix or [])])
        if self.doc_aware and int(self.span_len[-1]) <= longest:
            log(f"  WARNING: no document is longer than {longest} tokens; long draws fall back to the stream")

    def _episodes(self, spec, cache_dir, log):
        if not spec or not spec.get("path") or not float(spec.get("frac", 0)):
            return None
        from mercurius.data.episode_ds import EpisodeDataset
        return EpisodeDataset(spec["path"], self.tok, cache_dir=cache_dir, verbose=False)

    def longest(self):
        return max([self.cfg.seq] + [n for n, _ in (self.mix or [])])

    def _length(self, step):
        if self.mix and step >= self.cfg.length_mix_start:
            r, acc = float(torch.rand(1, generator=self.g_len)), 0.0
            for n, p in self.mix:
                acc += p
                if r <= acc:
                    return n
            return self.mix[-1][0]
        return self.cfg.seq

    def window(self, L, gen):
        if self.doc_aware:
            first = int(torch.searchsorted(self.span_len, torch.tensor(L + 1)))
            if first < len(self.span_len):
                j = first + int(torch.randint(len(self.span_len) - first, (1,), generator=gen))
                s, n = int(self.span_start[j]), int(self.span_len[j])
                o = s + int(torch.randint(n - L, (1,), generator=gen))
                return self.stream[o:o + L]
        o = int(torch.randint(len(self.stream) - L - 1, (1,), generator=gen))
        return self.stream[o:o + L]

    def next(self, step):
        """-> (ids (T,), supervision mask over the T-1 next-token targets or None, source name)."""
        L = self._length(step)
        u = float(torch.rand(1, generator=self.g_src))
        src = self.ep if (self.ep is not None and u < self.ep_frac) else \
            self.math if (self.math is not None and self.ep_frac <= u < self.ep_frac + self.math_frac) else None
        if src is not None:
            x, m = src.sample_with_mask(self.g_src, self.cfg.seq)
            if x is not None:
                name = "episodes" if src is self.ep else "math"
                self.counts[name] += 1
                return x, m[1:], name
        self.counts["text"] += 1
        return self.window(L, self.g_win), None, "text"

    def calibration_windows(self, n, seq, seed):
        """n windows in the training mix's proportions (for the MLA covariances), from their own generator."""
        g = torch.Generator().manual_seed(seed)
        out = []
        while len(out) < n:
            u = float(torch.rand(1, generator=g))
            src = self.ep if (self.ep is not None and u < self.ep_frac) else \
                self.math if (self.math is not None and self.ep_frac <= u < self.ep_frac + self.math_frac) else None
            if src is not None:
                x = src.sample(g, seq)
                if x is not None:
                    out.append(torch.as_tensor(x)[:seq])
                continue
            out.append(self.window(seq, g).clone())
        return out

    def state_dict(self):
        return {"len": self.g_len.get_state(), "win": self.g_win.get_state(), "src": self.g_src.get_state(),
                "counts": dict(self.counts)}

    def load_state_dict(self, d):
        self.g_len.set_state(d["len"])
        self.g_win.set_state(d["win"])
        self.g_src.set_state(d["src"])
        self.counts.update(d.get("counts", {}))


def eval_ids(path, tok, cache_dir=None):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cp = os.path.join(cache_dir, f"eval-{os.path.basename(path)}-{_key(path, tok)}.pt")
        if os.path.exists(cp):
            return torch.load(cp).long()
    ids = torch.tensor(tok(open(path, encoding="utf-8", errors="replace").read(),
                           add_special_tokens=False).input_ids, dtype=torch.int32)
    if cache_dir:
        torch.save(ids, cp)
    return ids.long()
