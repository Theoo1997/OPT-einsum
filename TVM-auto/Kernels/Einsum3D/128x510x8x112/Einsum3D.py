#!/usr/bin/env python3
"""
bench_einsum_mb.py
MetaSchedule-tuned einsum "nmk,bnk->mb" on CPU.

Loop shape
----------
M = 32, B = 126, N = 8, K = 256  (default)
C[m,b] = Σ_{n,k}  G[n,m,k] * X[b,n,k]
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
M, B, N, K = 128, 510, 8, 112
FLOPs = 2 * N * M * K * B     # mul-add per element

# ----------------------- CLI Arguments --------------------------------
cli = argparse.ArgumentParser()
cli.add_argument("--num-threads", type=int,
                 default=int(os.getenv("TVM_NUM_THREADS", 1)),
                 help="TVM_NUM_THREADS value")
cli.add_argument("--trials", type=int, default=512,
                 help="MetaSchedule tuning trials")
args = cli.parse_args()

NUM_THREADS = args.num_threads
TRIALS      = args.trials
os.environ["TVM_NUM_THREADS"] = str(NUM_THREADS)
NUM_CORES = max(1, NUM_THREADS // 2)   # assume 2 threads / core

# ------------------- 1. TE -> PrimFunc workload -----------------------
def einsum_mb(n=N, m=M, k=K, b=B):
    G = te.placeholder((n, m, k), name="G", dtype="float32")   # n m k
    X = te.placeholder((b, n, k), name="X", dtype="float32")   # b n k
    rn = te.reduce_axis((0, n), name="rn")
    rk = te.reduce_axis((0, k), name="rk")
    Out = te.compute(
        (m, b),
        lambda mm, bb: te.sum(G[rn, mm, rk] * X[bb, rn, rk], axis=[rn, rk]),
        name="Out",
    )
    return te.create_prim_func([G, X, Out])

# ------------------- 2. MetaSchedule tuning ---------------------------
TARGET  = f"llvm -mcpu=cascadelake -num-cores={NUM_CORES}"
WORKDIR = Path(__file__).parent / "ms_log_einsum_mb"
WORKDIR.mkdir(exist_ok=True)

print(f"[tune] target={TARGET}  trials={TRIALS}")
database =  tune_tir(
    mod=einsum_mb(),
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
with database:
    sch    = tvm.tir.Schedule(einsum_mb())
    module = tvm.build(sch.mod, target=TARGET)

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
g_np = np.random.randn(N, M, K).astype("float32")
x_np = np.random.randn(B, N, K).astype("float32")
out_np = np.zeros((M, B), dtype="float32")

g, x, out = [tvm.nd.array(x, dev) for x in (g_np, x_np, out_np)]

evaluator = module.time_evaluator(
    func_name=module.entry_name,
    dev=dev,
    number=50,
    repeat=3,
)
lat = np.min(evaluator(g, x, out).results)

# correctness with NumPy
ref = np.einsum("nmk,bnk->mb", g_np, x_np)
assert np.allclose(out.numpy(), ref, rtol=1e-3, atol=1e-3)
print("Correctness  : ✅  (TVM output matches NumPy)")

print("\n✅ Done — einsum benchmark complete.")
