"""Distillation of a converted model toward its teacher(s), with quantization-aware training for the deployed format.

Trainables (everything else frozen):
    VeRA vectors d, b on the configured projections         lr train.adapter_lr
    MLA latent maps (down, up_k, up_v) and the RoPE key      lr train.lr   (dense: they absorb the truncation)
    norm gains (ScaleNorm scalars, q_norm, gated norms,      lr train.lr   (fp32: a bf16 gain rounds a 2e-4 step away)
      final norm) and the GDN-2 decay vectors A_log, dt_bias
    per-head query maps                                      lr train.qmap_lr (identity init; projected onto the
                                                                           RoPE commutant after every step)
Optimizer NAdamW (or AdamW), OneCycle schedule, gradient clipping. Resumes exactly: weights, optimizer moments,
schedule, TAID state, data generators, QAT state, with a fingerprint of the data files.
"""
from __future__ import annotations

import json
import math
import os
import time

import types

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mercurius import guard
from mercurius.adapters import VeRALinear, inject_vera
from mercurius.arch.attention import MLAAttention
from mercurius.arch.common import ScaleNorm, rotary_dim, text_config, trunk
from mercurius.arch.gdn2 import GatedDeltaNet2
from mercurius.arch.model import load_model, load_tokenizer
from mercurius.convert.mla import project_rope_commute_, tie_rope_norm_deltas_
from mercurius.quant.qat import install_qat, release_caches
from mercurius.teachers import ServerTeacher, build_teachers
from .data import TrainingMix, eval_ids
from .losses import TAIDSchedule, chunk_terms


def _cast_forward(self, x):
    """Linear with an fp32 master weight, computed in the activation dtype."""
    return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


def setup_trainables(model, tcfg, log=print):
    for p in model.parameters():
        p.requires_grad_(False)
    a = tcfg.adapters
    inject_vera(model, a.targets, a.rank, a.d_init, seed=0, log=log)
    groups = {"dense": [], "adapter": [], "qmap": []}
    for m in model.modules():
        if isinstance(m, VeRALinear):
            for p in (m.vera_d, m.vera_b):
                p.requires_grad_(True)
                groups["adapter"].append(p)
        elif isinstance(m, MLAAttention):
            for lin in (m.down, m.up_k, m.up_v, m.k_rope):
                lin.weight.data = lin.weight.data.float()         # fp32 master, bf16 compute
                lin.weight.requires_grad_(True)
                lin.forward = types.MethodType(_cast_forward, lin)
                groups["dense"].append(lin.weight)
            if m.q_map is not None:
                m.q_map.requires_grad_(True)
                groups["qmap"].append(m.q_map)
        elif isinstance(m, GatedDeltaNet2) and tcfg.train_decay:
            for p in (m.A_log, m.dt_bias):
                p.data = p.data.float()
                p.requires_grad_(True)
                groups["dense"].append(p)
    if tcfg.train_norms:
        for name, m in model.named_modules():
            w = getattr(m, "weight", None)
            if isinstance(w, torch.nn.Parameter) and (isinstance(m, ScaleNorm) or name.endswith("norm")
                                                      or name.endswith("q_norm")):
                if w.requires_grad:
                    continue
                w.data = w.data.float()
                w.requires_grad_(True)
                groups["dense"].append(w)
    n = {k: sum(p.numel() for p in v) for k, v in groups.items()}
    log(f"  trainable: {n['dense'] / 1e6:.2f} M dense, {n['adapter'] / 1e6:.2f} M VeRA, {n['qmap'] / 1e6:.2f} M "
        f"query maps")
    return groups


def _fingerprint(paths):
    return {p: [os.stat(p).st_size, int(os.stat(p).st_mtime)] for p in paths if p and os.path.exists(p)}


class Trainer:
    def __init__(self, cfg, log=print):
        self.cfg, self.log = cfg, log
        self.out = cfg.run.out_dir
        self.dev = cfg.run.device
        self.ck = os.path.join(self.out, "ckpt")
        os.makedirs(self.ck, exist_ok=True)
        t = cfg.train
        if t.guards.mem_cap_gb:
            guard.cap_cuda_memory(t.guards.mem_cap_gb)
        torch.manual_seed(cfg.run.seed)
        conv = os.path.join(self.out, "converted")
        if not os.path.exists(os.path.join(conv, "config.json")):
            raise SystemExit(f"no converted model at {conv}: run `mercurius convert` first")
        self.model = load_model(conv, cfg.student.dtype, self.dev)
        self.tok = load_tokenizer(conv)
        self.tcfg = text_config(self.model)
        self.rd = rotary_dim(self.tcfg)
        self.trunk = trunk(self.model)
        self.groups = setup_trainables(self.model, t, log)
        if t.grad_checkpoint:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.config.use_cache = False
        self._ckpt_on = t.grad_checkpoint
        self.norm_ref = {n: p.detach().clone() for n, p in self.model.named_parameters()
                         if p.requires_grad and n.endswith("q_norm.weight")}
        cache = os.path.join(self.out, "cache")
        self.mix = TrainingMix(cfg.data, self.tok, cache, cfg.run.seed, log)
        self.eval_ids = eval_ids(cfg.data.eval_text, self.tok, cache) if cfg.data.eval_text else None
        self.teachers = build_teachers(cfg, self.out, log)
        if not self.teachers:
            raise SystemExit("no teacher configured")
        V = self.model.get_output_embeddings().weight.shape[0]
        for tch in self.teachers:
            if tch.head.shape[0] < V:
                raise SystemExit(f"teacher {tch.name}: vocabulary {tch.head.shape[0]} < student {V}")
            if isinstance(tch, ServerTeacher):
                tch.check(self.mix.stream[:64].tolist(), self.mix.longest() + 8)
        lrs = {"dense": t.lr, "adapter": t.adapter_lr, "qmap": t.qmap_lr}
        pg = [{"params": v, "lr": lrs[k]} for k, v in self.groups.items() if v]
        self.max_lr = [g["lr"] for g in pg]
        if t.optimizer == "nadamw":
            self.opt = torch.optim.NAdam(pg, betas=tuple(t.betas), weight_decay=t.weight_decay,
                                         decoupled_weight_decay=True)
        else:
            self.opt = torch.optim.AdamW(pg, betas=tuple(t.betas), weight_decay=t.weight_decay)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.opt, max_lr=self.max_lr, total_steps=max(t.steps, 1),
                                                         pct_start=t.pct_start, cycle_momentum=False)
        self.end = t.anneal["to"] if t.anneal else t.steps
        self.taid = (TAIDSchedule(self.end, t.taid.start, t.taid.end, t.taid.alpha, t.taid.beta)
                     if t.taid.enabled else None)
        self.qat_on = False
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(t.amp)
        self.scaler = torch.amp.GradScaler("cuda", enabled=t.amp == "fp16" and "cuda" in self.dev)
        self.step = 0
        self.hist = {"loss": [], "div": [], "data": [], "eval": []}
        self.data_paths = [cfg.data.text, (cfg.data.episodes or {}).get("path"), (cfg.data.math or {}).get("path")]
        res = os.path.join(self.ck, "resume.pt")
        if os.path.exists(res):
            self.load_resume(res)
        self._anneal_lr0 = None
        if t.anneal:
            le = self.sched.last_epoch
            self.sched.last_epoch = t.anneal["from"]
            self._anneal_lr0 = [float(x) for x in self.sched.get_lr()]
            self.sched.last_epoch = le

    # -------------------------------------------------------------------------------------------- state
    def trainable_state(self):
        return {n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad}

    def save_resume(self):
        blob = {"step": self.step, "weights": self.trainable_state(), "opt": self.opt.state_dict(),
                "sched": self.sched.state_dict(), "taid": self.taid.state_dict() if self.taid else None,
                "mix": self.mix.state_dict(), "qat": self.qat_on, "hist": self.hist,
                "scaler": self.scaler.state_dict(),
                "rng": {"torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                "data_fp": _fingerprint(self.data_paths)}
        p = os.path.join(self.ck, "resume.pt")
        torch.save(blob, p + ".tmp")
        os.replace(p + ".tmp", p)                        # an interrupted save never destroys the last resume file

    def load_resume(self, path):
        b = torch.load(path, map_location="cpu", weights_only=False)
        if b.get("data_fp") and b["data_fp"] != _fingerprint(self.data_paths):
            raise SystemExit("resume: the training data changed since the resume file was written")
        if b["sched"].get("total_steps") not in (None, max(self.cfg.train.steps, 1)):
            raise SystemExit(f"resume: written with train.steps {b['sched']['total_steps']}; the schedule depends "
                             f"on it, so continue with the same value")
        own = dict(self.model.named_parameters())
        with torch.no_grad():
            for k, v in b["weights"].items():
                own[k].copy_(v.to(own[k].dtype))
        self.opt.load_state_dict(b["opt"])
        self.sched.load_state_dict(b["sched"])
        if self.taid and b.get("taid"):
            self.taid.load_state_dict(b["taid"])
        self.mix.load_state_dict(b["mix"])
        if b.get("scaler"):
            self.scaler.load_state_dict(b["scaler"])
        torch.set_rng_state(b["rng"]["torch"])
        if b["rng"].get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(b["rng"]["cuda"])
        self.hist = b.get("hist", self.hist)
        self.step = int(b["step"])
        if b.get("qat"):
            self.install_qat()
        self.log(f"  resumed at step {self.step}")

    def save_trainables(self, tag):
        p = os.path.join(self.ck, f"trainable-{tag}.pt")
        torch.save(self.trainable_state(), p)
        return p

    # -------------------------------------------------------------------------------------------- pieces
    def install_qat(self):
        if not self.qat_on:
            install_qat(self.model, self.cfg.qat, self.log)
            self.qat_on = True

    def _set_checkpointing(self, T):
        t = self.cfg.train
        if not t.grad_checkpoint:
            return
        want = T > t.ckpt_above
        if want != self._ckpt_on:
            if want:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                self.model.gradient_checkpointing_disable()
            self.model.config.use_cache = False
            self._ckpt_on = want

    def loss(self, ids, sup):
        t = self.cfg.train
        x = ids.to(self.dev)
        T = x.shape[0]
        h_t = [(tch.hidden(x), tch.head, tch.weight) for tch in self.teachers]
        for h, _, _ in h_t:
            if h.shape[0] != T:
                raise RuntimeError("teacher / student length mismatch")
        h_s = self.trunk(input_ids=x.view(1, -1)).last_hidden_state[0]
        W_s = self.model.get_output_embeddings().weight
        lam = self.taid.t if self.taid else 1.0
        tot, n = torch.zeros(2, device=self.dev), 0
        nxt = x[1:]
        for i in range(0, T - 1, t.chunk):
            j = min(i + t.chunk, T - 1)
            if sup is not None and not bool(sup[i:j].any()):
                continue
            tc = [(h[i:j], W, w) for h, W, w in h_t]
            terms = checkpoint(chunk_terms, h_s[i:j], W_s, tc, nxt[i:j], t.divergence, lam,
                               t.taid.space, self.cfg.teacher_combine, t.reverse_weight, use_reentrant=False)
            if sup is not None:
                w = sup[i:j].to(self.dev, terms.dtype)
                tot = tot + (terms * w).sum(-1)
                n += int(w.sum())
            else:
                tot = tot + terms.sum(-1)
                n += j - i
        if n == 0:                                       # nothing supervised in this window
            z = h_s.float().sum() * 0
            return z, z.detach(), z.detach()
        div, data = tot[0] / n, tot[1] / n
        return div + t.ce_beta * data, div, data

    @torch.no_grad()
    def evaluate(self):
        if self.eval_ids is None:
            return {}
        self.model.eval()
        W = self.model.get_output_embeddings().weight
        row = {"step": self.step}
        amp = torch.autocast(self.dev.split(":")[0], dtype=self.amp_dtype or torch.bfloat16,
                             enabled=self.amp_dtype is not None)
        for L in self.cfg.train.eval_lengths:
            nll, cnt = 0.0, 0
            for s in range(0, min(len(self.eval_ids) - 1, 4 * L), L):
                x = self.eval_ids[s:s + L + 1].to(self.dev)
                if len(x) < 2:
                    break
                with amp:
                    h = self.trunk(input_ids=x[:-1].view(1, -1)).last_hidden_state[0]
                for i in range(0, h.shape[0], 1024):
                    lg = (h[i:i + 1024].to(W.dtype) @ W.T).float()
                    nll += float(torch.nn.functional.cross_entropy(lg, x[1 + i:1 + i + 1024], reduction="sum"))
                cnt += h.shape[0]
            row[L] = math.exp(nll / max(cnt, 1))
        self.model.train()
        release_caches(self.model)
        self.hist["eval"].append(row)
        self.log(f"  [eval @ {self.step}] " + "  ".join(f"ppl@{L} {row[L]:.3f}" for L in self.cfg.train.eval_lengths))
        return row

    # -------------------------------------------------------------------------------------------- loop
    def run(self):
        t, q = self.cfg.train, self.cfg.qat
        if self.step == 0:
            self.evaluate()
        self.model.train()
        t0, seen = time.time(), 0
        while self.step < self.end:
            step = self.step + 1
            if q.enabled and not self.qat_on and step >= max(q.start_step, 1):
                self.install_qat()
                self.evaluate()                          # the quantized model before any QAT step = the PTQ cost
                self.model.train()
            if guard.mem_available_gb() < t.guards.min_avail_gb:
                self.log(f"  stopping at step {self.step}: available memory below {t.guards.min_avail_gb} GiB "
                         f"(resume continues from here)")
                self.save_resume()
                return 3
            guard.wait_until_cool(t.guards.gpu_temp_pause, t.guards.gpu_temp_resume, 200, 200, log=self.log)
            ids, sup, src = self.mix.next(step)
            self._set_checkpointing(len(ids))
            with torch.autocast(self.dev.split(":")[0], dtype=self.amp_dtype or torch.bfloat16,
                                enabled=self.amp_dtype is not None):
                loss, div, data = self.loss(ids, sup)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_([p for g in self.groups.values() for p in g], t.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.sched.step()
            self.opt.zero_grad(set_to_none=True)
            if self._anneal_lr0 is not None and step + 1 >= t.anneal["from"]:
                f = min(1.0, (step + 1 - t.anneal["from"]) / max(t.anneal["to"] - t.anneal["from"], 1))
                for g, l0 in zip(self.opt.param_groups, self._anneal_lr0):
                    g["lr"] = l0 * 0.5 * (1 + math.cos(math.pi * f))
            project_rope_commute_(self.model, self.rd)
            tie_rope_norm_deltas_(self.model, self.rd, self.norm_ref)
            if self.taid:
                self.taid.update(step, float(div.detach()))
            self.step = step
            seen += len(ids)
            self.hist["loss"].append(float(loss.detach()))
            self.hist["div"].append(float(div.detach()))
            self.hist["data"].append(float(data.detach()))
            if step % t.log_every == 0:
                el = time.time() - t0
                c = self.mix.counts
                tot = max(sum(c.values()), 1)
                peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
                self.log(f"  step {step}/{self.end}  loss {float(loss):.4f}  div {float(div):.4f}  "
                         f"data {float(data):+.4f}  taid {self.taid.t if self.taid else 1:.3f}  "
                         f"mix {'/'.join(f'{k[0]}{100 * v // tot}' for k, v in c.items())}  "
                         f"{seen / el:.0f} tok/s  peak {peak:.1f} GiB")
            if t.save_every and step % t.save_every == 0:
                self.save_resume()
            if t.eval_every and step % t.eval_every == 0:
                self.evaluate()
                if t.keep_step_ckpts:
                    self.save_trainables(f"step{step}")
        self.evaluate()
        self.save_resume()
        p = self.save_trainables("final")
        with open(os.path.join(self.out, "train_log.json"), "w") as f:
            json.dump(self.hist, f, default=str)
        self.log(f"  done: {p}")
        return 0
