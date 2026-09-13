#!/usr/bin/env bash
# Refuse to proceed if free space is below the floor. Filling ext4 root
# to zero on a Jetson can corrupt it into needing a full reflash.
FLOOR_GB=${FLOOR_GB:-5}
avail=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$avail" -lt "$FLOOR_GB" ]; then
  echo "DISK GUARD: only ${avail}GB free, floor is ${FLOOR_GB}GB — refusing." >&2
  exit 1
fi
echo "disk guard OK: ${avail}GB free"
