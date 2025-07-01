

cd TVM-auto/third_party/tvm

# Initialize and update all submodules (including libbacktrace)
git submodule update --init --recursive
```bash
sudo apt update
sudo apt install -y \
    git cmake build-essential libtinfo-dev zlib1g-dev \
    python3-dev python3-pip libedit-dev libxml2-dev \
    libllvm15 llvm-15-dev libssl-dev gcc-11 g++-11
'''    

# Use llvm-14-dev or similar if 15 is unavailable on your distro.
