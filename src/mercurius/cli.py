"""mercurius: convert a pretrained hybrid model, distill it back with quantization-aware training, export it.

    mercurius run      CONFIG [key=value ...]    convert (if not done) -> train -> export
    mercurius convert  CONFIG [key=value ...]    surgery + calibration -> <out_dir>/converted
    mercurius train    CONFIG [key=value ...]    distillation + QAT (resumes automatically)
    mercurius export   CONFIG [key=value ...] [--ckpt PATH]
    mercurius eval     MODEL_DIR --text FILE [--lengths 2048,8192]
    mercurius info     MODEL_DIR

CONFIG is a YAML file (see configs/); any value can be overridden as dotted.key=value.
"""
from __future__ import annotations

import argparse
import os
import sys


def _log(msg):
    print(msg, flush=True)


def _cfg(a):
    from mercurius.config import load_config, save_config
    cfg = load_config(a.config, a.overrides)
    os.makedirs(cfg.run.out_dir, exist_ok=True)
    save_config(cfg, os.path.join(cfg.run.out_dir, "config.resolved.json"))
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mercurius", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "convert", "train", "export"):
        p = sub.add_parser(c)
        p.add_argument("config")
        p.add_argument("overrides", nargs="*")
        if c == "export":
            p.add_argument("--ckpt", default=None)
    p = sub.add_parser("eval")
    p.add_argument("model")
    p.add_argument("--text", required=True)
    p.add_argument("--lengths", default="2048,8192")
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("info")
    p.add_argument("model")
    a = ap.parse_args(argv)

    if a.cmd == "info":
        import json
        spec = json.load(open(os.path.join(a.model, "config.json"))).get("mercurius")
        if not spec:
            print("not a Mercurius checkpoint")
            return 1
        kinds = {}
        for s in spec["layers"].values():
            for k, v in s.items():
                if isinstance(v, dict):
                    kinds[v.get("kind", k)] = kinds.get(v.get("kind", k), 0) + 1
        print(json.dumps({"modules": kinds, "kv_values_per_token": spec["kv_values_per_token"],
                          "kv_cache": spec.get("kv_cache"), "quantized_tensors": len(spec.get("quant", {}))},
                         indent=1))
        return 0
    if a.cmd == "eval":
        from mercurius.eval_ppl import perplexity
        res = perplexity(a.model, a.text, [int(x) for x in a.lengths.split(",")], a.device)
        print("  ".join(f"ppl@{k} {v:.3f}" for k, v in res.items()))
        return 0

    cfg = _cfg(a)
    if a.cmd in ("convert", "run"):
        if a.cmd == "convert" or not os.path.exists(os.path.join(cfg.run.out_dir, "converted", "config.json")):
            from mercurius.convert.pipeline import convert
            convert(cfg, _log)
        if a.cmd == "convert":
            return 0
    if a.cmd in ("train", "run"):
        from mercurius.train.trainer import Trainer
        rc = Trainer(cfg, _log).run()
        if rc or a.cmd == "train":
            return rc
    if a.cmd in ("export", "run"):
        from mercurius.export import export
        export(cfg, getattr(a, "ckpt", None), _log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
