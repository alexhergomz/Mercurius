"""The teacher as a separate llama.cpp process, queried for hidden states.

WHY NOT IN-PROCESS. D5 routes the training-time teacher to 35B-A3B and is
blocked on loader work: Qwen3_5MoeExperts holds gate_up_proj/down_proj as raw
3-D nn.Parameter, so bnb.Linear4bit has nothing to wrap. That blocker only binds
if the teacher must live inside our autograd graph -- and it need not. The
teacher is frozen, no gradient flows through it, and D13 already established that
what training consumes is its FINAL HIDDEN STATE, not its logits. llama.cpp
loads the same GGUF and returns exactly that tensor.

WHY NOT A CACHE. D13 caches h_t, and D12's cache is indexed by a flat offset
which forces `align = cache["seq"]`, i.e. deterministic sequential batching. That
conflicts with sample_with_mask, which exists because diff targets sit at the END
of long prompts and naive windows drop them. Worse, 73 episodes of 1,973 run to
65k tokens and hold 35% of the corpus, so any precomputed window gives the
teacher DIFFERENT context from the student's and the targets stop corresponding.
Querying live sends exactly the ids the student saw, so the context matches by
construction, and for a 150-step run it is also cheaper: a step is ~58 s at 140
tok/s, the call adds ~8.6 s for 8192 tokens (+15%), while removing an in-process
teacher forward that D12 measured at ~69% of a step.

VERIFIED before use:
  * the endpoint accepts TOKEN IDS, and the result is bit-identical to sending
    the detokenised text (max diff 0.00e+00), so there is no round-trip through
    text and no retokenisation drift on student-generated continuations;
  * llama.cpp's tokenizer matches ours exactly (1007/1007 ids on calib_mix);
  * the returned vector is the post-output_norm, pre-lm_head state: decoding it
    through the head dequantised from the same GGUF gives coherent continuations.

Flags that are load-bearing on the server:
    --pooling none        per token, not one pooled vector
    --embd-normalize -1   RAW. The default L2-normalises to unit length, which
                          silently destroys the scale the loss reads. Observed
                          norms are ~108; `check()` refuses anything near 1.0.
    -ub >= max window     PHYSICAL batch, not context. Embeddings are computed
                          in ONE physical batch, so a window longer than -ub is
                          refused with HTTP 500 ("input (N tokens) is too large
                          to process. increase the physical batch size"). -c
                          being large enough is NOT sufficient: training at
                          seq 8192 against -ub 4096 fails mid-run, after the
                          model has loaded and the first eval has passed.
    --parallel 1          one slot gets the whole context rather than -c/n_slots.

    llama-server -m <gguf> --embeddings --pooling none --embd-normalize -1 \
        -ngl 99 -c 16384 -b 16384 -ub 16384 --parallel 1 --port 8077
"""
import json
import time
import urllib.error
import urllib.request

import torch


class TeacherServer:
    def __init__(self, url="http://127.0.0.1:8077", timeout=1800, retries=3, patience=600):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.patience = patience
        self.calls = 0
        self.seconds = 0.0
        self.tokens = 0

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        # Retry for up to `patience` seconds, not 3 quick attempts: a teacher restart takes
        # 60-90 s (model load), and 12 s of retries turned every restart into a training
        # crash. HTTP 4xx are request errors and are NOT retried forever.
        last = None
        deadline = time.time() + self.patience
        attempt = 0
        while True:
            try:
                return json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
            except urllib.error.HTTPError as e:
                last = e
                if 400 <= e.code < 500 or attempt >= self.retries - 1 and time.time() > deadline:
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = e
                if time.time() > deadline:
                    break
            attempt += 1
            time.sleep(min(30, 2 * attempt))
        raise RuntimeError(f"teacher server {self.url} unreachable: {last}")

    def hidden(self, ids, device="cuda", dtype=torch.float32):
        """Final hidden state for a 1-D LongTensor / list of token ids -> (L, d_t)."""
        if torch.is_tensor(ids):
            ids = ids.detach().flatten().tolist()
        d = self._post("/embeddings", {"input": [int(t) for t in ids]})
        if isinstance(d, dict):
            d = d.get("data", [d])
        t0 = time.time()
        h = torch.tensor(d[0]["embedding"], dtype=dtype, device=device)
        self.calls += 1
        self.tokens += h.shape[0]
        self.seconds += time.time() - t0
        if h.shape[0] != len(ids):
            # llama.cpp may drop/add a boundary token; the caller aligns on the
            # shorter length rather than silently pairing mismatched positions
            pass
        return h

    def check(self, ids, max_len=None):
        """Fail loudly at startup rather than train on silently wrong targets.

        `max_len` probes the LONGEST window the run can produce. That is not
        --seq: an on-policy step generates from x[:, :plen] and appends up to
        --on-policy-gen tokens, and the anchor may sit within 8 tokens of the
        end, so a window can reach seq + on_policy_gen. Measured the hard way --
        a run died four minutes in when an 8,429-token window met -ub 8192.
        Embeddings are computed in ONE physical batch, so -ub must cover that
        maximum; -c being large enough is not sufficient.
        """
        if max_len:
            probe = (list(ids) * (max_len // max(len(ids), 1) + 1))[:max_len]
            try:
                hp = self.hidden(probe, device="cpu")
            except RuntimeError as e:
                raise SystemExit(
                    f"teacher server rejected a {max_len}-token window, which "
                    f"this run CAN produce (seq + on-policy-gen). Restart it "
                    f"with -b and -ub at least {max_len}: {e}")
            if hp.shape[0] != max_len:
                raise SystemExit(
                    f"teacher returned {hp.shape[0]} states for a {max_len}-token "
                    f"probe; alignment would be wrong at full length")
        h = self.hidden(ids[:64], device="cpu")
        n = h.norm(dim=-1).mean().item()
        if abs(n - 1.0) < 0.05:
            raise SystemExit(
                "teacher server returned L2-NORMALISED embeddings (mean norm "
                f"{n:.3f}). Restart it with --embd-normalize -1, or every target "
                "is wrong by a per-token scale factor.")
        if h.shape[0] != min(64, len(ids)):
            raise SystemExit(
                f"teacher returned {h.shape[0]} states for {min(64,len(ids))} ids; "
                "token alignment is not one-to-one and targets would be shifted.")
        return n
