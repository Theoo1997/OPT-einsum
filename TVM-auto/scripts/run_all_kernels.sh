#!/usr/bin/env bash
# run_all_kernels.sh
#
# Walks through every kernel → size directory and executes the matching
# Python benchmark with TVM_NUM_THREADS = 1 2 4 8.

set -euo pipefail

# ----- list your top-level kernel folders here ------------------------
KERNELS=( "MMM" "MVM" "Kronecker" "Einsum3D" "Einsum4D" "Fw_Convolution" )

# ----- colour helper (bold cyan) -------------------------------------
banner () {
  local txt="$1"
  echo -e "\n\033[1;36m${txt}\033[0m"
}

cd ../Kernels
# ----- main loop ------------------------------------------------------
for kernel in "${KERNELS[@]}"; do
  if [[ ! -d "$kernel" ]]; then
    echo "Warning: directory '$kernel' not found, skipping." >&2
    continue
  fi

  for size_dir in "$kernel"/*/ ; do
    [[ -d "$size_dir" ]] || continue      # skip if not a directory

    size_name=$(basename "$size_dir")
    driver_py="${size_dir%/}/${kernel}.py"

    if [[ ! -f "$driver_py" ]]; then
      echo "Warning: ${driver_py} not found, skipping." >&2
      continue
    fi

    for t in 1 2 4 8; do
      banner "KERNEL: ${kernel} | SIZE: ${size_name} | TVM_NUM_THREADS=${t}"
      python "$driver_py" --num-threads "$t" --trials 1000
    done
  done
done
