"""Resource guards for a SHARED machine: memory and temperature.

This board is a GB10 with 121 GB of UNIFIED memory shared by CPU and GPU, and
it is not ours alone -- vLLM (--gpu-memory-utilization 0.35, ~42 GB), a Triton
speech server and others run on it too. On 2026-09-21 a 4B/27B run exhausted
the unified pool at its first eval and the machine hard-reset without logging
anything; the kernel had been reporting NVRM NV_ERR_NO_MEMORY for half an hour.
Unified-memory exhaustion does not produce a Python OOM, it produces a freeze.
The user also reported the box running hot.

So the trainer (1) caps its own CUDA allocator, turning an overshoot into a
catchable torch OOM, (2) refuses to continue below a floor of system
MemAvailable, and (3) pauses when the GPU or any ACPI zone runs hot. All three
act only on this process. Nothing here ever kills anything else.
"""
import glob
import subprocess
import time

import torch


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return float("nan")


def cap_cuda_memory(cap_gb):
    """Hard ceiling on this process's CUDA allocator. Returns the fraction set."""
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    frac = min(1.0, cap_gb / total)
    torch.cuda.set_per_process_memory_fraction(frac)
    return frac, total


def gpu_temp_c():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return float("nan")


def acpi_max_c():
    vals = []
    for z in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            vals.append(int(open(z).read()) / 1000.0)
        except Exception:
            pass
    return max(vals) if vals else float("nan")


def temps():
    return gpu_temp_c(), acpi_max_c()


def wait_until_cool(gpu_pause, gpu_resume, acpi_pause, acpi_resume, log=print):
    """Block while too hot. Returns seconds spent waiting."""
    g, a = temps()
    if not (g >= gpu_pause or a >= acpi_pause):
        return 0.0
    t0 = time.time()
    log(f"  [thermal] GPU {g:.0f}C / ACPI max {a:.0f}C over the limit "
        f"({gpu_pause}/{acpi_pause}); pausing until {gpu_resume}/{acpi_resume}")
    while g > gpu_resume or a > acpi_resume:
        time.sleep(15)
        g, a = temps()
    log(f"  [thermal] resumed after {time.time() - t0:.0f}s at GPU {g:.0f}C / "
        f"ACPI max {a:.0f}C")
    return time.time() - t0


def cool_to(gpu_target, acpi_target, log=print):
    """Wait until BOTH readings are at or below the targets. Used before long
    uninterrupted GPU bursts (an 8192-token eval forward of the 27B), which
    add ~15C with no step boundary to pause at: on 2026-09-21 the eval took
    the GPU from under 78C to 85C, where the external monitor stops the run."""
    g, a = temps()
    if g <= gpu_target and a <= acpi_target:
        return 0.0
    t0 = time.time()
    while g > gpu_target or a > acpi_target:
        time.sleep(10)
        g, a = temps()
    log(f"  [thermal] pre-burst cool-down {time.time() - t0:.0f}s -> GPU {g:.0f}C "
        f"/ ACPI max {a:.0f}C")
    return time.time() - t0


class ThermalPacer:
    """Pause INSIDE a forward/backward when the GPU runs hot.

    Step-boundary checks are too coarse at long context: one 8192-token step
    (27B teacher forward + 4B student forward/backward with recompute) runs
    tens of seconds uninterrupted, and on 2026-09-21 took the GPU from under
    the 78C pause to 84C, one degree from the external kill line. A forward
    pre-hook on every decoder layer checks the temperature through NVML (5 us
    per read) and sleeps until cool. Under gradient checkpointing the hooks
    also fire during backward's recomputation, so backward is paced too.

    Sleeping only stops Python from enqueuing work; the GPU drains what is
    already queued (milliseconds at this granularity) and then idles. It
    changes timing, never results.
    """

    def __init__(self, pause_c, resume_c, acpi_pause_c=None, acpi_resume_c=None,
                 log=print):
        import pynvml
        pynvml.nvmlInit()
        self._nv = pynvml
        self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.pause_c, self.resume_c = pause_c, resume_c
        self.acpi_pause_c, self.acpi_resume_c = acpi_pause_c, acpi_resume_c
        self.log = log
        self.paused_s, self.n_pauses = 0.0, 0
        self._last_acpi, self._acpi = 0.0, 0.0
        self._handles = []

    def gpu(self):
        return self._nv.nvmlDeviceGetTemperature(self._h, self._nv.NVML_TEMPERATURE_GPU)

    def power_w(self):
        try:
            return self._nv.nvmlDeviceGetPowerUsage(self._h) / 1000.0
        except Exception:
            return float("nan")

    def acpi(self):
        now = time.time()
        if now - self._last_acpi > 0.5:       # sysfs is slower; 2 Hz is plenty
            self._acpi, self._last_acpi = acpi_max_c(), now
        return self._acpi

    def check(self, *_):
        g = self.gpu()
        a = self.acpi() if self.acpi_pause_c else 0.0
        if g < self.pause_c and (not self.acpi_pause_c or a < self.acpi_pause_c):
            return
        t0 = time.time()
        while g > self.resume_c or (self.acpi_resume_c and a > self.acpi_resume_c):
            time.sleep(0.5)
            g = self.gpu()
            a = acpi_max_c() if self.acpi_pause_c else 0.0
        self.paused_s += time.time() - t0
        self.n_pauses += 1

    def attach(self, model):
        """Hook every decoder layer of `model`. Returns the number hooked."""
        from mercurius.surgery.norm_fusion import get_trunk
        n = 0
        for layer in get_trunk(model).layers:
            self._handles.append(layer.register_forward_pre_hook(self.check))
            n += 1
        return n

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
