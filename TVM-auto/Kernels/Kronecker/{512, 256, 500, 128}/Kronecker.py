#!/usr/bin/env python3
"""
bench_kron_einsum.py
MetaSchedule-tuned kernel for

    Y[d,e] = Σ_{a,b}  A[d,a] * X[a,b] * B[b,e]

Loop order reference:
for d in D:
  for e in E:
    Y[d,e] = 0
for a in A:
  for b in B:
    Y[d,e] += B[b,e] * A[d,a] * X[a,b]
"""

# --------------------------- Imports ----------------------------------
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
import argparse, os

# ----------------------- Default Problem Sizes ------------------------
D, E = 512, 128   # output dims
A, B = 500, 128   # reduction dims  (B here is 128)
FLOPs = 2 * A * B * D * E

# ----------------------- CLI Arguments --------------------------------
cli = argparse.ArgumentParser()
cli.add_argument("--num-threads", type=int,
                 default=int(os.getenv("TVM_NUM_THREADS", 8)),
                 help="TVM_NUM_THREADS value")
cli.add_argument("--trials", type=int, default=1,
                 help="MetaSchedule tuning trials")
args = cli.parse_args()

NUM_THREADS = args.num_threads
TRIALS      = args.trials
os.environ["TVM_NUM_THREADS"] = str(NUM_THREADS)
NUM_CORES   = max(1, NUM_THREADS // 2)   # assume 2 HW threads / core

# ------------------- 1. TE -> PrimFunc workload -----------------------
def kron_einsum(a=A, b=B, d=D, e=E):
    A_mat = te.placeholder((d, a), name="A_mat", dtype="float32")  # D × A
    B_mat = te.placeholder((b, e), name="B_mat", dtype="float32")  # B × E
    X     = te.placeholder((a, b), name="X",     dtype="float32")  # A × B

    ra = te.reduce_axis((0, a), name="ra")
    rb = te.reduce_axis((0, b), name="rb")

    Y = te.compute(
        (d, e),
        lambda dd, ee: te.sum(A_mat[dd, ra] * X[ra, rb] * B_mat[rb, ee],
                              axis=[ra, rb]),
        name="Y",
    )
    return te.create_prim_func([A_mat, B_mat, X, Y])

# ------------------- 2. MetaSchedule tuning ---------------------------
TARGET  = f"llvm -mcpu=cascadelake -num-cores={NUM_CORES}"
WORKDIR = Path(__file__).parent / "ms_log_kron_einsum"
WORKDIR.mkdir(exist_ok=True)

print(f"[tune] target={TARGET}  trials={TRIALS}")
tune_tir(
    mod=kron_einsum(),
    target=Target(TARGET),
    work_dir=str(WORKDIR),
    max_trials_global=TRIALS,
    max_trials_per_task=TRIALS,
    num_trials_per_iter=8,
    builder=Builder.create("local"),
    runner=Runner.create(
        "local",
        evaluator_config=ms_config.EvaluatorConfig(number=20, repeat=3,
                                                   enable_cpu_cache_flush=True)
    ),
    cost_model=XGBModel(),
    strategy=ms.search_strategy.ReplayTrace(),
)

# ------------------- 3. Best tuning record ----------------------------
db = JSONDatabase(work_dir=str(WORKDIR))
best_rec = min(
    db.get_all_tuning_records(),
    key=lambda r: np.mean([float(x.value) if isinstance(x, FloatImm) else float(x)
                           for x in r.run_secs])
)
sch = Schedule(best_rec.workload.mod)
best_rec.trace.apply_to_schedule(sch, False)

# ------------------- 4. Build tuned module ----------------------------
target = Target(TARGET)
build_res, = Builder.create("local", max_workers=1).build([BuilderInput(sch.mod, target)])
assert build_res.error_msg is None
artifact = build_res.artifact_path

# ------------------- 5. Measure with Runner ---------------------------
mod_clean = RemoveWeightLayoutRewriteBlock()(sch.mod)
args_info = ArgInfo.from_entry_func(mod_clean)
runner = Runner.create(
    "local",
    max_workers=1,
    evaluator_config=ms_config.EvaluatorConfig(number=100, repeat=1, min_repeat_ms=1),
    timeout_sec=100
)
run_res = runner.run([RunnerInput(artifact, "llvm", args_info)])[0].result()
assert run_res.error_msg is None

latencies = [float(x.value) if isinstance(x, FloatImm) else float(x)
             for x in run_res.run_secs]
best_time = min(latencies)
gflops    = FLOPs / best_time / 1e9

print("\nRaw run_secs:", latencies)
print(f"Min time   : {best_time:.6f} s")
print(f"GFLOP/s    : {gflops:.2f}")

# ------------------- 6. Correctness check -----------------------------
dev  = tvm.cpu()
A_np = np.random.randn(D, A).astype("float32")
B_np = np.random.randn(B, E).astype("float32")
X_np = np.random.randn(A, B).astype("float32")
Y_np = np.zeros((D, E), dtype="float32")

module = tvm.build(sch.mod, target=TARGET)
A_t, B_t, X_t, Y_t = [tvm.nd.array(arr, dev) for arr in (A_np, B_np, X_np, Y_np)]

_ = module.time_evaluator(module.entry_name, dev, number=10, repeat=1)(A_t, B_t, X_t, Y_t)

ref = np.einsum("da,ab,be->de", A_np, X_np, B_np)      
assert np.allclose(Y_t.numpy(), ref, rtol=1e-1, atol=1e-1)
print("Correctness  : ✅  matches NumPy")

print("\n✅ Done — Kronecker einsum benchmark complete.")
