# OPT-einsum 📐⚡

Reproducible artifact for the **“OPT-Einsum: Auto-tuned & Hand-Optimised Tensor Contractions”**.

This repo contains three independent baselines:

| Directory   | Description                                    |
|-------------|------------------------------------------------|
| **OneDNN/** | Intel OneDNN implementation used as a CPU baseline |
| **TVM-auto/** | Auto-tuned kernel built with Apache TVM MetaSchedule |
| **Proposed/** | Our own hand-optimised AVX2 version |

The Artifact Evaluation (AE) committee can rebuild everything with one command.

---

## Quick-start (local machine)

```bash
git clone --recursive https://github.com/Theoo1997/OPT-einsum.git
cd OPT-einsum
conda env create -f environment.yml
conda activate OPT-einsum
