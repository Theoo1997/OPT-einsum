#!/usr/bin/env python3
"""
bench_conv1x1.py
MetaSchedule-tuned 1×1 convolution (NHWC) on CPU:

out[b,y,x,m] = Σ_d  in[b,y,x,d] * filter[m,d]
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

# ------------------------ Default Problem sizes -----------------------
Batch, H, W = 30, 28, 28   # spatial
M, D     = 256, 256  # in-channels, out-channels
FLOPs = 2 * Batch * H * W * M * D

# ------------------------ CLI arguments -------------------------------
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
NUM_CORES = max(1, NUM_THREADS // 2)      # assume 2 threads per core

# -------------------- 1. TE -> PrimFunc workload ----------------------
def conv1x1(b=Batch, y=H, x=W, d=D, m=M):
    Input  = te.placeholder((b, y, x, d), name="Input",  dtype="float32")
    Filter = te.placeholder((m, d),       name="Filter", dtype="float32")
    rd = te.reduce_axis((0, d), name="rd")
    Output = te.compute(
        (b, y, x, m),
        lambda bb, yy, xx, mm: te.sum(Input[bb, yy, xx, rd] * Filter[mm, rd], axis=rd),
        name="Output"
    )
    return te.create_prim_func([Input, Filter, Output])

# -------------------- 2. MetaSchedule tuning --------------------------
TARGET  = f"llvm -mcpu=cascadelake -num-cores={NUM_CORES}"
WORKDIR = Path(__file__).parent / "ms_log_conv1x1"
WORKDIR.mkdir(exist_ok=True)

print(f"[tune] target={TARGET}  trials={TRIALS}")
tune_tir(
    mod=conv1x1(),
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

# -------------------- 3. Extract best schedule ------------------------
db = JSONDatabase(work_dir=str(WORKDIR))
best_rec = min(
    db.get_all_tuning_records(),
    key=lambda r: np.mean([float(x.value) if isinstance(x, FloatImm) else float(x)
                           for x in r.run_secs])
)
sch = Schedule(best_rec.workload.mod)
best_rec.trace.apply_to_schedule(sch, False)

# -------------------- 4. Build tuned module ---------------------------
target = Target(TARGET)
build_res, = Builder.create("local", max_workers=1).build([BuilderInput(sch.mod, target)])
assert build_res.error_msg is None
artifact = build_res.artifact_path

# -------------------- 5. Measure with Runner --------------------------
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

latencies = [float(x.value) if isinstance(x, FloatImm) else float(x) for x in run_res.run_secs]
best_time = min(latencies)
gflops    = FLOPs / best_time / 1e9

print("\nRaw run_secs:", latencies)
print(f"Min time   : {best_time:.6f} s")
print(f"GFLOP/s    : {gflops:.2f}")

# -------------------- 6. Correctness check ----------------------------
dev = tvm.cpu()
inp_np  = np.random.randn(Batch, H, W, D).astype("float32")
filt_np = np.random.randn(M, D).astype("float32")
out_np  = np.zeros((Batch, H, W, M), dtype="float32")

module = tvm.build(sch.mod, target=TARGET)
inp_t, filt_t, out_t = [tvm.nd.array(arr, dev) for arr in (inp_np, filt_np, out_np)]

_ = module.time_evaluator(module.entry_name, dev, number=10, repeat=1)(inp_t, filt_t, out_t)

# NumPy reference
ref = np.einsum("byxd,md->byxm", inp_np, filt_np)
assert np.allclose(out_t.numpy(), ref, rtol=1e-3, atol=1e-3)
print("Correctness  : ✅  matches NumPy")

print("\n✅ Done — 1×1 convolution benchmark complete.")
