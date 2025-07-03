#!/usr/bin/env python3
"""
bench_matvec_runbest.py
MetaSchedule‑tuned 768 × 768 matrix–vector multiply (MVM) with accurate Builder/Runner timing.

C (J,) = A (J × K) · B (K,)
"""

# ---------------------------- Imports --------------------------------
from pathlib import Path
import numpy as np
import tvm
from tvm import te, meta_schedule as ms
from tvm.meta_schedule.tir_integration import tune_tir
from tvm.meta_schedule.builder import Builder, BuilderInput
from tvm.meta_schedule.runner import Runner, RunnerInput, config as ms_config
from tvm.meta_schedule.arg_info import ArgInfo
from tvm.meta_schedule.database import JSONDatabase
from tvm.meta_schedule.cost_model import XGBModel
from tvm.tir import FloatImm, Schedule
from tvm.tir.transform import RemoveWeightLayoutRewriteBlock
from tvm.target import Target
import argparse, os, time

# --------------------- Problem Dimensions ---------------------------
J, K = 768, 50257
FLOPs = 2 * J * K

# --------------------- CLI Arguments --------------------------------
p = argparse.ArgumentParser()
p.add_argument("--num-threads", type=int, default=int(os.getenv("TVM_NUM_THREADS", 1)),
               help="TVM_NUM_THREADS value")
p.add_argument("--trials", type=int, default=10, help="Tuning trials")
args = p.parse_args()

NUM_THREADS = args.num_threads
TRIALS = args.trials
os.environ["TVM_NUM_THREADS"] = str(NUM_THREADS)
NUM_CORES = max(1, NUM_THREADS // 2)

# --------------------- Workload Definition --------------------------
def matvec(j=J, k=K):
    A = te.placeholder((j, k), name="A", dtype="float32")
    B = te.placeholder((k,), name="B", dtype="float32")
    red_k = te.reduce_axis((0, k), name="rk")
    C = te.compute(
        (j,),
        lambda jj: te.sum(A[jj, red_k] * B[red_k], axis=red_k),
        name="C",
    )
    return te.create_prim_func([A, B, C])
# --------------------- MetaSchedule Tuning --------------------------
TARGET = f"llvm -mcpu=cascadelake -num-cores={NUM_CORES}"
WORKDIR = Path(__file__).parent / "ms_log_matvec"
WORKDIR.mkdir(exist_ok=True)

print(f"[tune] target   : {TARGET}")
print(f"[tune] trials   : {TRIALS}")
start = time.time()

tune_tir(
    mod=matvec(),
    target=Target(TARGET),
    work_dir=str(WORKDIR),
    max_trials_global=TRIALS,
    max_trials_per_task=TRIALS,
    num_trials_per_iter=8,
    builder=Builder.create("local"),
    runner=Runner.create(
        "local",
        evaluator_config=ms_config.EvaluatorConfig(number=20, repeat=3, enable_cpu_cache_flush=True)
    ),
    cost_model=XGBModel(),
    strategy=ms.search_strategy.ReplayTrace(),
)
print(f"[tune] finished in {time.time() - start:.1f}s")

# --------------------- Load Best Schedule ---------------------------
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
        continue
if not valid:
    raise RuntimeError("No valid timing data in records.")

best_record = min(valid, key=lambda t: t[1])[0]
sch = Schedule(best_record.workload.mod)
best_record.trace.apply_to_schedule(sch, remove_postproc=False)

# --------------------- Build Final Module ---------------------------
target = Target(TARGET)
builder = Builder.create("local", max_workers=1)
(builder_result,) = builder.build([BuilderInput(sch.mod, target)])
assert builder_result.error_msg is None
artifact = builder_result.artifact_path

# --------------------- Run Final Kernel -----------------------------
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

# --------------------- Correctness Test -----------------------------
dev = tvm.cpu()
a_np = np.random.randn(J, K).astype("float32")
b_np = np.random.randn(K).astype("float32")
c_np = np.zeros(J, dtype="float32")

print("[build] tvm.build with tuned schedule …")
module = tvm.build(sch.mod, target=TARGET)

dev = tvm.cpu()                       # or tvm.cuda(0) for GPU
a_tvm = tvm.nd.array(a_np, dev)
b_tvm = tvm.nd.array(b_np, dev)
c_tvm = tvm.nd.array(np.zeros(J, dtype="float32"), dev)

evaluator = module.time_evaluator(
    func_name=module.entry_name,      # entry of generated module
    dev=dev,
    number=50,                        # 50 inner runs
    repeat=3,                         # 3 measurements
)
times = evaluator(a_tvm, b_tvm, c_tvm).results

# correctness check
ref = a_np @ b_np
assert np.allclose(c_tvm.numpy(), ref, rtol=1e-3, atol=1e-3)
print("Correctness : ✅ matches NumPy")
