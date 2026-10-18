"""Fast measurement harness.

The earlier gates reloaded the model 4-6 times per script; on this board that is
minutes of pure overhead per run. Two fixes:

  * reference logits are cached to disk, keyed by model path + probe
  * weight-only variants restore from a CPU state-dict snapshot instead of
    reloading from disk

Structural variants (Stage B swaps modules) still need a rebuild, but only once
each rather than once per measurement.
"""
import hashlib, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE = "probe/cache"
os.makedirs(CACHE, exist_ok=True)

PROBE_TEXTS = [
    "The Jetson AGX Orin has unified memory shared between CPU and GPU.",
    "In linear attention, the delta rule updates a fixed-size state matrix.",
    "def quicksort(a):\n    if len(a) <= 1:\n        return a",
    "Cuando el modelo pierde la codificacion posicional, las capas lineales",
]


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / a.norm().clamp_min(1e-12)).item()


class Harness:
    def __init__(self, model_path, texts=None, max_length=48, dtype=torch.float32,
                 device="cuda"):
        self.path, self.dtype, self.device = model_path, dtype, device
        self.texts = texts or PROBE_TEXTS
        self.tok = AutoTokenizer.from_pretrained(model_path)
        b = self.tok(self.texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=max_length)
        self.batch = {k: v.to(device) for k, v in b.items()}
        key = hashlib.sha256(
            (model_path + json.dumps(self.texts) + str(max_length) + str(dtype)
             ).encode()).hexdigest()[:16]
        self.ref_file = os.path.join(CACHE, f"ref-{key}.pt")
        self._snapshot = None

    # ---------------------------------------------------------------- loading
    def load(self):
        m = AutoModelForCausalLM.from_pretrained(
            self.path, dtype=self.dtype, device_map=self.device)
        return m.eval()

    @torch.no_grad()
    def logits(self, model):
        return model(**self.batch).logits.float().clone()

    def reference(self, model=None):
        """Reference logits, cached to disk across runs."""
        if os.path.exists(self.ref_file):
            return torch.load(self.ref_file, map_location=self.device)
        own = model is None
        m = model or self.load()
        ref = self.logits(m)
        torch.save(ref.cpu(), self.ref_file)
        if own:
            del m
            torch.cuda.empty_cache()
        return ref.to(self.device)

    # -------------------------------------------------------------- snapshots
    def snapshot(self, model):
        """Keep a CPU copy so weight-only variants can be undone without reload."""
        self._snapshot = {k: v.detach().to("cpu", copy=True)
                          for k, v in model.state_dict().items()}
        return self._snapshot

    def restore(self, model):
        if self._snapshot is None:
            raise RuntimeError("no snapshot taken")
        model.load_state_dict({k: v.to(self.device)
                               for k, v in self._snapshot.items()}, strict=False)
        return model

    # ------------------------------------------------------------ measurement
    def measure(self, tag, model, ref, verbose=True):
        new = self.logits(model)
        d = (ref - new).abs()
        r = rel(ref, new)
        t1 = (ref.argmax(-1) == new.argmax(-1)).float().mean().item() * 100
        if verbose:
            print(f"  {tag:<40} max {d.max().item():.3e}  relL2 {r:.3e}  "
                  f"top1 {t1:7.3f}%", flush=True)
        return {"tag": tag, "max": d.max().item(), "rel": r, "top1": t1}
