import shutil, sys
def require_free_gb(floor=5.0):
    free = shutil.disk_usage("/").free / 2**30
    if free < floor:
        sys.exit(f"DISK GUARD: {free:.1f}GB free, floor {floor}GB — refusing.")
    return free
