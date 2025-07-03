# PC Setup 🛠️🧠

This guide prepares your local Linux machine for running and reproducing the OPT-einsum experiments.

Before running `TVM-auto/scripts/setup_tvm.sh`, you must install a few system dependencies and set up the Conda environment.

---

## 🧱 System dependencies (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y \
    git cmake build-essential libtinfo-dev zlib1g-dev \
    python3-dev python3-pip libedit-dev libxml2-dev \
    libssl-dev gcc-11 g++-11 libllvm15 llvm-15-dev
```

## Clone TVM submodules
TVM requires several third-party components (e.g., libbacktrace, dmlc-core).

Make sure you fetch all submodules:
```bash
cd TVM-auto/third_party/tvm
git submodule update --init --recursive
```

▶️ Run the build script
Once the environment and submodules are ready, build TVM with:

```bash
bash ./scripts/setup_tvm.sh
```


▶️ Produce the auto-tuing results
Once the environment and submodules are ready, build TVM with:

```bash
bash ./scripts/run_all_kernels.sh
```