#!/usr/bin/env python3
"""
bench_einsum_mbr.py
MetaSchedule-tuned einsum "rnmk,bnk->mbr" on CPU.

Loop shape:
Y[m,b,r] = Σ_{n,k} G[r,n,m,k] * X[b,n,k]
"""

# ---------------------------- Imports ---------------------------------
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

# ----------------------- Default Problem Sizes ------------------------
N, B, R, M, K = 32, 510, 32, 8, 100
FLOPs = 2 * R * N * M * K * B

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
NUM_CORES = max(1, NUM_THREADS // 2)   # assume 2 threads / core

# ------------------- 1. TE -> PrimFunc workload -----------------------
def einsum_mbr(r=R, n=N, m=M, k=K, b=B):
    G = te.placeholder((r, n, m, k), name="G", dtype="float32")   # r n m k
    X = te.placeholder((b, n, k), name="X", dtype="float32")      # b n k
    rn = te.reduce_axis((0, n), name="rn")
    rk = te.reduce_axis((0, k), name="rk")
    Y = te.compute(
        (m, b, r),
        lambda mm, bb, rr: te.sum(G[rr, rn, mm, rk] * X[bb, rn, rk], axis=[rn, rk]),
        name="Y",
    )
    return te.create_prim_func([G, X, Y])

# ------------------- 2. MetaSchedule tuning ---------------------------
TARGET  = f"llvm -mcpu=cascadelake -num-cores={NUM_CORES}"
WORKDIR = Path(__file__).parent / "ms_log_einsum_mbr"
WORKDIR.mkdir(exist_ok=True)

print(f"[tune] target={TARGET}  trials={TRIALS}")
database =  tune_tir(
    mod=einsum_mbr(),
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

# ------------------- 3. Get best tuning record ------------------------
db = JSONDatabase(work_dir=str(WORKDIR))
best_rec = min(
    db.get_all_tuning_records(),
    key=lambda r: np.mean([float(x.value) if isinstance(x, FloatImm) else float(x)
                           for x in r.run_secs])
)
sch = Schedule(best_rec.workload.mod)
best_rec.trace.apply_to_schedule(sch, False)

# ------------------- 4. Build tuned schedule --------------------------
target = Target(TARGET)
builder = Builder.create("local", max_workers=1)
(build_res,) = builder.build([BuilderInput(sch.mod, target)])
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
(run_fut,) = runner.run([RunnerInput(artifact, "llvm", args_info)])
run_res = run_fut.result()
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
g_np = np.random.randn(R, N, M, K).astype("float32")
x_np = np.random.randn(B, N, K).astype("float32")
y_np = np.zeros((M, B, R), dtype="float32")

print("[build] tvm.build with tuned schedule …")
module = tvm.build(sch.mod, target=TARGET)

g = tvm.nd.array(g_np, dev)
x = tvm.nd.array(x_np, dev)
y = tvm.nd.array(y_np, dev)

evaluator = module.time_evaluator(
    func_name=module.entry_name,
    dev=dev,
    number=50,
    repeat=3,
)
_ = evaluator(g, x, y)  # timing not used here

ref = np.einsum("rnmk,bnk->mbr", g_np, x_np)
assert np.allclose(y.numpy(), ref, rtol=1e-3, atol=1e-3)
print("Correctness  : ✅  (TVM output matches NumPy)")

print("\n✅ Done — einsum mbr benchmark complete.")
