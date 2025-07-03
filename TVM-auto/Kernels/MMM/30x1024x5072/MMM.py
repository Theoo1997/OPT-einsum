#!/usr/bin/env python3
"""
bench_matmul_runbest.py
MetaSchedule‑tuned 30 × 1024 × 1024 GEMM with accurate Builder/Runner timing.

C (I × J) = A (I × K) · B (K × J)
"""
# ---------------- Imports ---------------------------------------------
from pathlib import Path
import numpy as np
import tvm
from tvm import te, meta_schedule as ms
from tvm.meta_schedule.tir_integration import tune_tir
from tvm.meta_schedule.builder import Builder, BuilderInput
from tvm.meta_schedule.runner import Runner, RunnerInput, config as ms_config
from tvm.meta_schedule.arg_info import ArgInfo
from tvm.meta_schedule.database import JSONDatabase
from tvm.tir import FloatImm, Schedule
from tvm.tir.transform import RemoveWeightLayoutRewriteBlock
from tvm.target import Target
from tvm.meta_schedule.cost_model import XGBModel 
import argparse, os, time

# ---------------- Matrix sizes & tuning budget ------------------------
I, J, K = 30, 1024, 5072
FLOPs = 2 * I * J * K

p = argparse.ArgumentParser()
p.add_argument("--num-threads", type=int,
               default=int(os.getenv("TVM_NUM_THREADS", 1)),
               help="physical cores for target & runtime")
p.add_argument("--trials", type=int, default=10)
               
args = p.parse_args()
NUM_THREADS = args.num_threads
TRIALS = args.trials
os.environ["TVM_NUM_THREADS"] = str(NUM_THREADS)

if NUM_THREADS == 1: 
     NUM_CORES = NUM_THREADS/2 # our CPU has 2 threads per core
else:
    NUM_CORES = 1 
# ---------------- 1. Define MatMul Workload ---------------------------
def matmul(i=I, k=K, j=J):
    A = te.placeholder((i, k), name="A", dtype="float32")
    B = te.placeholder((k, j), name="B", dtype="float32")
    kk = te.reduce_axis((0, k), name="kk")
    C = te.compute((i, j), lambda ii, jj: te.sum(A[ii, kk] * B[kk, jj], axis=kk), name="C")
    return te.create_prim_func([A, B, C])

# ---------------- 2. Tune with MetaSchedule ---------------------------
TARGET = f"llvm -mcpu=cascadelake -num-cores={NUM_CORES}"
WORKDIR = Path(__file__).parent / "ms_log_matmul"
WORKDIR.mkdir(exist_ok=True)

print(f"[tune] target   : {TARGET}")
print(f"[tune] trials   : {TRIALS}")

start = time.time()
tune_tir(
    mod=matmul(),
    target=Target(TARGET),
    work_dir=str(WORKDIR),
    max_trials_global=TRIALS,
    max_trials_per_task=TRIALS,
    num_trials_per_iter=8,
    builder=Builder.create("local"),
    runner=Runner.create("local", evaluator_config=ms_config.EvaluatorConfig(number=20, repeat=3, enable_cpu_cache_flush=True)),
    cost_model=XGBModel(),
    strategy=ms.search_strategy.ReplayTrace(),
)
print(f"[tune] finished in {time.time() - start:.1f}s")

# ---------------- 3. Load Best Tuning Record --------------------------
db = JSONDatabase(work_dir=str(WORKDIR))
records = db.get_all_tuning_records()
if not records:
    raise RuntimeError("No tuning records found.")

valid = []
for rec in records:
    try:
        secs = [float(x.value) if isinstance(x, FloatImm) else float(x) for x in rec.run_secs]
        if secs:
            valid.append((rec, np.mean(secs)))
    except Exception:
        pass
if not valid:
    raise RuntimeError("No valid timing data in records.")

best_record = min(valid, key=lambda t: t[1])[0]
sch = Schedule(best_record.workload.mod)
best_record.trace.apply_to_schedule(sch, remove_postproc=False)

# ---------------- 4. Build the Schedule -------------------------------
target = Target(TARGET)
builder = Builder.create("local", max_workers=1)
(builder_result,) = builder.build([BuilderInput(sch.mod, target)])
assert builder_result.error_msg is None
artifact = builder_result.artifact_path

# ---------------- 5. Run the Compiled Module --------------------------
mod_clean = RemoveWeightLayoutRewriteBlock(skip_ndarray_rewrite=True)(sch.mod)
args_info = ArgInfo.from_entry_func(mod_clean)

runner = Runner.create(
    "local",
    max_workers=1, 
    evaluator_config=ms_config.EvaluatorConfig(number=100, repeat=1, min_repeat_ms=1),
    timeout_sec=100
)
(runner_future,) = runner.run([RunnerInput(artifact, "llvm", args_info)])
runner_result = runner_future.result()
assert runner_result.error_msg is None

latencies = [float(x.value) if isinstance(x, FloatImm) else float(x) for x in runner_result.run_secs]
min_time = min(latencies)
gflops = FLOPs / min_time / 1e9

print("\nRaw run_secs:", latencies)
print(f"Min time   : {min_time:.6f} s")
print(f"GFLOP/s    : {gflops:.2f}")
