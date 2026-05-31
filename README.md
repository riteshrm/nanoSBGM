# nanoSBGM

Minimal PyTorch implementations of **score-based generative modeling (SBGM)** with:

* **VE SDE** (Variance Exploding)
* **VP SDE** (Variance Preserving)
* **sub-VP SDE**

and their corresponding **probability-flow ODE (PF-ODE)** samplers, using a lightweight DiT-based score network.

## Features

* Minimal and readable implementations
* Three common SDE parameterizations (VE / VP / sub-VP)
* Score matching training objective
* Stochastic sampling via Euler–Maruyama
* Deterministic PF-ODE sampling
* Equation-aligned code
* Educational focus

## Repository Structure

```text
nanoSBGM/
├── VESDE.py
├── VPSDE.py
├── subVPSDE.py
└── model.py
```

## Theory and Derivations

Detailed explanations and mathematical derivations are available in the accompanying blog post:

* [Score Matching Notes](https://riteshrm.github.io/posts/score-matching/)

The notes cover:

* Score matching objective
* SDE formulation and reverse-time dynamics
* Probability-flow ODE (PF-ODE)
* Sampling intuition

## Data

Download the dataset (Hugging Face) into `butterflies/`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="riteshrm/butterflies", repo_type="dataset", local_dir="butterflies"
)
```

The training scripts use `torchvision.datasets.ImageFolder`, so `butterflies/` should be laid out like:

```
butterflies/
  class_0/
  class_1/
  ...
```

## Usage

Train with a specific SDE:

```bash
python VESDE.py
# or
python VPSDE.py
# or
python subVPSDE.py
```

Note: the scripts currently assume `NUM_CLASSES = 5` and assert it matches the number of folders found under `butterflies/`.

During training, the script periodically:

- Saves sample grids from the **SDE** sampler as `sample_epoch_sde_*.png`
- Saves sample grids from the **PF-ODE** sampler as `sample_epoch_ode_*.png`
- Saves checkpoints as `dit_conditional_epoch_*.pth`

Most hyperparameters (image size, model size, batch size, number of steps, etc.) are defined at the top of each script.

## Credits

The DiT backbone implementation in `model.py` is adapted from:

* https://github.com/sayakpaul/nanoDiT
