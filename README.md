# nanoSBGM

Minimal **score-based generative modeling (SBGM)** experiments in PyTorch with **three SDE parameterizations** and their corresponding **probability-flow ODE (PF-ODE)** samplers.

Accompanying notes (derivations + intuition): https://riteshrm.github.io/posts/score-matching/

## What’s implemented

This repo contains three standalone training scripts (each script defines the SDE, loss, and both samplers):

- **VE SDE** (Variance Exploding): `VESDE.py`
- **VP SDE** (Variance Preserving): `VPSDE.py`
- **sub-VP SDE**: `subVPSDE.py`

Each script provides:

- A denoising score-matching style objective (`loss(...)`)
- An SDE sampler (`sample(...)`, Euler–Maruyama)
- A PF-ODE sampler (`sample_pf_ode(...)`, deterministic)

## Data

Download the dataset (Hugging Face) into `butterflies/`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="riteshrm/butterflies", repo_type="dataset", local_dir="butterflies"
)
```

The training scripts use `torchvision.datasets.ImageFolder`, so the directory should be laid out like:

```
butterflies/
  class_0/
  class_1/
  ...
```

## Run

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

## Model

The backbone is a tiny DiT-style model in `model.py`.

## Credits

- `model.py` is adapted from from [sayakpaul/nanoDiT](https://github.com/sayakpaul/nanoDiT).
