"""Validate a queued arm's FULL command line without running any work.

WHY THIS EXISTS. Six times in this project a run has launched with a config that
was not what the arm claimed (#17 five dropped flags; #22 tau non-persistent so
F2A2 was disabled; #23 the build() detector keyed on a deleted tensor; _is_f
matching every mlp.gate_proj; xatlu.alpha landing in the dense group; and the
--mla-groups override that could silently have trained 8094 at 4096). Every one
was discoverable in seconds by parsing the args and LOOKING. Argparse accepts a
typo'd value for a flag it does not know only if you never check, so check.

It works by composing the real command line through the arm script, then
monkeypatching ArgumentParser.parse_args to raise after a genuine parse -- so
train.py builds its full parser and validates everything, and then stops before
touching a GPU.

    .venv/bin/python experiments/dry_parse_arms.py \
        experiments/run_c0_arm.sh "gate150 --mla-gate xatlu"
"""
import argparse
import shlex
import subprocess
import sys


class Parsed(Exception):
    def __init__(self, ns):
        self.ns = ns


def argv_for(script, extra):
    """Run the arm script with `python -m ...train` replaced by `echo`."""
    r = subprocess.run(
        ["bash", "-c",
         "sed 's|^exec .venv/bin/python -m mercurius.recovery.train|exec echo CMD|' "
         f"{shlex.quote(script)} > /tmp/_dry_arm.sh && "
         f"bash /tmp/_dry_arm.sh {extra}"],
        capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith("CMD")]
    if not lines:
        raise SystemExit(f"could not compose a command from {script}:\n{r.stdout}\n{r.stderr}")
    return shlex.split(lines[0][4:])


SHOW = ("tag", "dial", "steps", "seq", "mla_groups", "mla_dc", "lr", "vera_lr",
        "latent_ext_lr", "mla_gate", "mla_taps", "mla_conv", "mla_conv_where",
        "mla_mol", "mla_mol_latent", "mla_mol_routed", "mla_mol_struct", "mol_struct_E", "mol_struct_routing", "math_data", "math_frac", "mla_calib", "mla_budget", "episodes", "mol_pairing", "mol_calib_tokens", "mol_router", "mol_init", "mol_bias_gamma", "mol_fisher_windows", "mol_var_coef", "mol_scale", "mol_spread", "mol_balance", "f2a2", "synth_data",
        "train_data", "episode_frac", "keep_step_ckpts", "eval_every")


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    script, extra = sys.argv[1], " ".join(sys.argv[2:])
    real = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = \
        lambda self, a=None, n=None: (_ for _ in ()).throw(Parsed(real(self, a, n)))
    from mercurius.recovery import train
    sys.argv = ["train"] + argv_for(script, extra)
    try:
        train.main()
    except Parsed as p:
        for k in SHOW:
            if hasattr(p.ns, k):
                print(f"  {k:<18} {getattr(p.ns, k)}")
        return
    except SystemExit as e:
        raise SystemExit(f"ARGPARSE REJECTED IT (exit {e.code}) -- fix before queueing")
    raise SystemExit("parse_args was never reached; train.py's structure changed")


if __name__ == "__main__":
    main()
