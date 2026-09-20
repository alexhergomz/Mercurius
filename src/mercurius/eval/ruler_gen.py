"""RULER synthetic NIAH task generation, reimplemented against the published spec.

Source: NVIDIA/RULER (Hsieh et al., arXiv:2404.06654), scripts/data/synthetic/
niah.py + constants.py + synthetic.yaml, vendored under third_party/ruler/ so
the templates here can be diffed against the originals.

Why reimplement rather than run the harness: RULER's prediction path targets a
vLLM / TRT-LLM / OpenAI-compatible server. This model is a hand-converted
checkpoint that only exists as a local nn.Module on a Jetson with a hand-built
torch, and pip-resolving that harness risks replacing the torch build. Task
GENERATION is pure python and fully specified, so it is copied exactly; only
scoring is ours, and it is a superset of RULER's.

Faithful: needle string, task template, answer_prefix, haystack types
(noise/essay/needle), key/value types (adjective-noun words, 7-digit numbers,
uuid4), the 40-point depth grid, the singular/plural template rewrite, the
binary search that fits the haystack to a token budget, and string_match_all.

Deviations, all recorded because they break exact comparability with published
RULER numbers:
  - haystack essays come from sgoel9/paul_graham_essays (215 essays, 510,677
    words); RULER ships a PaulGrahamEssays.json that is not in the public tree.
  - the tokenizer is this model's own (vocab 248,320), so token budgets differ
    from any other model's at the same nominal length. That is unavoidable and
    is also true of RULER itself across models.
Numbers from this file are "RULER-style", not official RULER scores.
"""
import json
import random
import re
import uuid
from pathlib import Path

import numpy as np
from nltk.tokenize import sent_tokenize
from wonderwords.random_word import _get_words_from_text_file
from mercurius.paths import DATA_DIR

ESSAY = Path(str(DATA_DIR / 'ruler' / 'PaulGrahamEssays.json'))

NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
TEMPLATE = ("Some special magic {type_needle_v} are hidden within the following "
            "text. Make sure to memorize it. I will quiz you about the "
            "{type_needle_v} afterwards.\n{context}\nWhat are all the special "
            "magic {type_needle_v} for {query} mentioned in the provided text?")
ANSWER_PREFIX = (" The special magic {type_needle_v} for {query} mentioned in "
                 "the provided text are")
NOISE = ("The grass is green. The sky is blue. The sun is yellow. Here we go. "
         "There and back again.")

# synthetic.yaml, retrieval family only. vt/cwe/fwe/qa are separate task files.
TASKS = {
    "niah_single_1":   dict(haystack="noise",  k="words", v="numbers", nk=1, nv=1, nq=1),
    "niah_single_2":   dict(haystack="essay",  k="words", v="numbers", nk=1, nv=1, nq=1),
    "niah_single_3":   dict(haystack="essay",  k="words", v="uuids",   nk=1, nv=1, nq=1),
    "niah_multikey_1": dict(haystack="essay",  k="words", v="numbers", nk=4, nv=1, nq=1),
    "niah_multikey_2": dict(haystack="needle", k="words", v="numbers", nk=1, nv=1, nq=1),
    "niah_multikey_3": dict(haystack="needle", k="uuids", v="uuids",   nk=1, nv=1, nq=1),
    "niah_multivalue": dict(haystack="essay",  k="words", v="numbers", nk=1, nv=4, nq=1),
    "niah_multiquery": dict(haystack="essay",  k="words", v="numbers", nk=1, nv=1, nq=4),
}
TOKENS_TO_GENERATE = 128
DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))

_WORDS = None
_HAYSTACK_WORDS = None


def _words():
    global _WORDS
    if _WORDS is None:
        nouns = _get_words_from_text_file("nounlist.txt")
        adjs = _get_words_from_text_file("adjectivelist.txt")
        _WORDS = sorted({f"{a}-{n}" for a in adjs for n in nouns})
    return _WORDS


def _essay_words():
    global _HAYSTACK_WORDS
    if _HAYSTACK_WORDS is None:
        txt = json.load(open(ESSAY))["text"]
        _HAYSTACK_WORDS = re.sub(r"\s+", " ", txt).split(" ")
    return _HAYSTACK_WORDS


def _rand(kind, rng):
    if kind == "numbers":
        return str(rng.randint(10 ** 6, 10 ** 7 - 1))
    if kind == "words":
        return rng.choice(_words())
    if kind == "uuids":
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))
    raise NotImplementedError(kind)


def build_one(cfg, num_haystack, rng, seed):
    """One sample. Mirrors niah.py:generate_input_output."""
    nk = max(cfg["nk"], cfg["nq"])
    keys, values, needles = [], [], []
    for _ in range(nk):
        keys.append(_rand(cfg["k"], rng))
        vals = []
        for _ in range(cfg["nv"]):
            vals.append(_rand(cfg["v"], rng))
            needles.append(NEEDLE.format(type_needle_v=cfg["v"], key=keys[-1],
                                         value=vals[-1]))
        values.append(vals)
    random.Random(seed).shuffle(needles)

    if cfg["haystack"] == "essay":
        hay = _essay_words()
        if num_haystack <= len(hay):
            text = " ".join(hay[:num_haystack])
        else:
            reps = (num_haystack + len(hay) - 1) // len(hay)
            text = " ".join((hay * reps)[:num_haystack])
        sents = sent_tokenize(text.strip())
        pos = ([0] + sorted(int(len(sents) * (d / 100))
                            for d in rng.sample(DEPTHS, len(needles)))
               + [len(sents)])
        parts = []
        for i in range(1, len(pos)):
            parts.append(" ".join(sents[pos[i - 1]:pos[i]]))
            if i - 1 < len(needles):
                parts.append(needles[i - 1])
        context = " ".join(parts)
    else:
        if cfg["haystack"] == "noise":
            sents = [NOISE] * num_haystack
        else:                                     # distractor needles
            sents = [NEEDLE.format(type_needle_v=cfg["v"],
                                   key=_rand(cfg["k"], rng),
                                   value=_rand(cfg["v"], rng))
                     for _ in range(num_haystack)]
        for idx, el in zip(sorted(rng.sample(range(num_haystack), len(needles)),
                                  reverse=True), needles):
            sents.insert(idx, el)
        context = "\n".join(sents)

    qi = rng.sample(range(nk), cfg["nq"])
    queries = [keys[i] for i in qi]
    answers = [a for i in qi for a in values[i]]
    query = (", ".join(queries[:-1]) + ", and " + queries[-1]
             if len(queries) > 1 else queries[0])

    tpl, pre, tnv = TEMPLATE, ANSWER_PREFIX, cfg["v"]
    if cfg["nq"] * cfg["nv"] == 1:
        # RULER passes template AND answer_prefix as ONE --template string and
        # rewrites that, so the prefix is singularised too: a single-needle
        # prompt ends "...provided text is", not "...are". Rewriting only the
        # body leaves the model completing an ungrammatical prompt, which shifts
        # the NLL of the very span being scored. replace() is local, so applying
        # it to each half matches applying it to the concatenation.
        def sing(s):
            return (s.replace("Some", "A").replace("are all", "is")
                     .replace("are", "is").replace("answers", "answer"))
        tpl, pre = sing(tpl), sing(pre)
        tnv = tnv[:-1]
    body = tpl.format(type_needle_v=tnv, context=context, query=query)
    prefix = pre.format(type_needle_v=tnv, query=query)
    return body, prefix, answers


def generate(task, max_seq_length, num_samples, tok, seed=42,
             tokens_to_generate=TOKENS_TO_GENERATE, verbose=False):
    """Fit the haystack to the budget by binary search, then emit samples."""
    cfg = TASKS[task]
    rng = random.Random(seed)
    incremental = 500 if cfg["haystack"] == "essay" else 25
    if cfg["haystack"] != "essay" and max_seq_length < 4096:
        incremental = 5

    def ntok(body, prefix):
        return len(tok(body + prefix).input_ids)

    b, p, _ = build_one(cfg, incremental, random.Random(seed), seed)
    per = ntok(b, p) / incremental
    lo, hi = incremental, max(int((max_seq_length / per) * 3), incremental * 2)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        b, p, _ = build_one(cfg, mid, random.Random(seed), seed)
        if ntok(b, p) + tokens_to_generate <= max_seq_length:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    num_haystack = best if best is not None else incremental
    if verbose:
        print(f"    {task}@{max_seq_length}: haystack={num_haystack} "
              f"({per:.1f} tok/unit)", flush=True)

    out = []
    for i in range(num_samples):
        used = num_haystack
        while True:
            body, prefix, answers = build_one(cfg, used, rng, seed + i)
            n = ntok(body, prefix) + tokens_to_generate
            if n <= max_seq_length or used <= incremental:
                break
            used -= incremental
        out.append({"task": task, "input": body, "answer_prefix": prefix,
                    "outputs": answers, "length": n})
    return out


def string_match_all(preds, refs):
    """RULER's metric: mean over samples of the fraction of golds present."""
    return round(sum(sum(1.0 if r.lower() in p.lower() else 0.0 for r in ref)
                     / len(ref) for p, ref in zip(preds, refs))
                 / len(preds) * 100, 2)
