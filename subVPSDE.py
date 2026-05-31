import torch
import torch.nn as nn
import torch.optim as optim
from model import NanoDiT
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

# --- Hyperparameters ---
NUM_CLASSES = 5  
IMG_SIZE = 64
IMG_CHANNELS = 3 
# DiT specific parameters
LATENT_DIM = 768
PATCH_SIZE = 2
MODEL_DEPTH = 2
MODEL_HEADS = 2

# Training parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
EPOCHS = 600
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPILE = True
AMP_DTYPE = torch.bfloat16 # automatic mixed-precision unless you wanna touch grass
# Sampling parameters
SAMPLE_INTERVAL = 1  # Sample every N epochs
NUM_SAMPLES_PER_CLASS = 2  # Number of images to sample per class during evaluation
CFG_SCALE = 5.0
NUM_STEPS=5
# Others
CHECKPOINT_SAVE_INTERVAL = 1
DATA_DIR = "butterflies" # Directory where the dataset is stored.

class subVPSDE(nn.Module):
    def __init__(self, beta_min, beta_max, eps):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.eps = eps
        self.score_model = NanoDiT(
                input_size=IMG_SIZE,
                patch_size=PATCH_SIZE,
                in_channels=IMG_CHANNELS,
                hidden_size=LATENT_DIM,
                depth=MODEL_DEPTH,
                num_heads=MODEL_HEADS,
                num_classes=NUM_CLASSES,
                timestep_freq_scale=1000,
            )
    
    def get_integrated_beta_t(self, t):
        return torch.tensor(self.beta_min*t + 0.5 * (self.beta_max - self.beta_min) * t**2).to(DEVICE)
    
    def get_beta_t(self, t):
        return torch.tensor(self.beta_min + (self.beta_max - self.beta_min) * t).to(DEVICE)

    def get_integrated_alpha_t(self, t):
        return torch.exp(-0.5 * self.get_integrated_beta_t(t))
    
    def get_integrated_sigma_t(self, t):
        return 1 - torch.exp(-self.get_integrated_beta_t(t))
    
    def perturbation_kernel(self, x_0, t, noise=None):
        alpha_t = self.get_integrated_alpha_t(t).view(-1, 1, 1, 1)
        sigma_t = self.get_integrated_sigma_t(t).view(-1, 1, 1, 1)
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = alpha_t * x_0 + sigma_t * noise
        return x_t, noise, sigma_t
    
    def sde(self, x, t):
        drift = -0.5 * self.get_beta_t(t).view(-1, 1, 1, 1) * x
        diffusion = torch.sqrt(self.get_beta_t(t)*(1-torch.exp(-2*self.get_integrated_beta_t(t)))).view(-1, 1, 1, 1)
        return drift, diffusion
    
    @torch.no_grad()
    def sample(self, target_classes_list, num_steps, num_samples_per_cls=1):
        """Generate images for specified target classes using CFG."""
        self.score_model.eval()
        num_target_cls = len(target_classes_list)
        total_images_to_sample = num_samples_per_cls * num_target_cls

        # Initial state
        x_t = torch.randn((total_images_to_sample, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), device=DEVICE)

        # Prepare conditional labels
        sample_cls_labels_list = []
        for c_idx in target_classes_list:
            sample_cls_labels_list.extend([c_idx] * num_samples_per_cls)
        conditional_labels = torch.tensor(sample_cls_labels_list, device=DEVICE).long()

        y = conditional_labels
        
        ts = torch.linspace(1, self.eps, num_steps).to(DEVICE)
        delta_t = torch.tensor(-1/num_steps).to(DEVICE)
        for t in ts:
            tin = torch.ones(x_t.shape[0], device=DEVICE) * t
            score = self.score_model(x_t, tin, y)
            drift, diffusion = self.sde(x_t, t)
            x_prevt = x_t + (drift - diffusion**2*score) * delta_t + diffusion * torch.sqrt(-delta_t)*torch.randn_like(x_t)
            x_t = x_prevt
        
        images = (x_t + 1) / 2.0  # De-normalize from [-1, 1] to [0, 1]
        images = torch.clamp(images, 0.0, 1.0)

        self.score_model.train() # Set model to train.
        return images, conditional_labels
    
    @torch.no_grad()
    def sample_pf_ode(self, target_classes_list, num_steps, num_samples_per_cls=1):
        """Generate images for specified target classes using CFG."""
        self.score_model.eval()
        num_target_cls = len(target_classes_list)
        total_images_to_sample = num_samples_per_cls * num_target_cls

        # Initial state
        x_t = torch.randn((total_images_to_sample, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), device=DEVICE)

        # Prepare conditional labels
        sample_cls_labels_list = []
        for c_idx in target_classes_list:
            sample_cls_labels_list.extend([c_idx] * num_samples_per_cls)
        conditional_labels = torch.tensor(sample_cls_labels_list, device=DEVICE).long()

        y = conditional_labels
        
        ts = torch.linspace(1, self.eps, num_steps)
        delta_t = torch.tensor(-1/num_steps).to(DEVICE)
        for t in ts:
            tin = torch.ones(x_t.shape[0], device=DEVICE) * t
            score = self.score_model(x_t, tin, y)
            drift, diffusion = self.sde(x_t, t)
            x_prevt = x_t + (drift - 0.5*diffusion**2*score) * delta_t
            x_t = x_prevt

        images = (x_t + 1) / 2.0  # De-normalize from [-1, 1] to [0, 1]
        images = torch.clamp(images, 0.0, 1.0)

        self.score_model.train() # Set model to train.
        return images, conditional_labels

    def loss(self, x_0, y):
        t = self.eps + (1- self.eps)*torch.rand((x_0.shape[0], )).to(DEVICE)
        x_t, noise, sigma_t = self.perturbation_kernel(x_0, t)
        s_theta = self.score_model(x_t, t, y)
        loss = F.mse_loss(s_theta, -noise/sigma_t, reduction='mean')
        return loss
    


sbgm = subVPSDE(beta_min=0.1, beta_max=20, eps=1e-5).to(DEVICE)

optimizer = optim.AdamW(sbgm.score_model.parameters(), lr=LEARNING_RATE)
scaler = torch.GradScaler() if AMP_DTYPE is not None else None
amp_context = (
    torch.autocast(device_type=torch.device(DEVICE).type, dtype=AMP_DTYPE) 
    if AMP_DTYPE is not None
    else nullcontext()
)
if AMP_DTYPE:
    print(f"Using automatic mixed-precision in {AMP_DTYPE} (change if needed).")

# --- Dataset and DataLoader  ---
ds_trfs = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ]
)
train_dataset = torchvision.datasets.ImageFolder(DATA_DIR, transform=ds_trfs)
train_classes = list(set(train_dataset.class_to_idx.values()))
assert NUM_CLASSES == len(train_classes)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,  # Adjust based on your system
    pin_memory=True,  # Useful when training on GPU
    drop_last=True,
    prefetch_factor=2, # Adjust based on your system
)

# --- Training Loop ---
print(f"Training on {DEVICE}")
print(f"Using custom model: {type(sbgm.score_model).__name__}")
print(f"Model Parameters: {sum(p.numel() for p in sbgm.score_model.parameters() if p.requires_grad)}")

for epoch in range(EPOCHS):
    sbgm.score_model.train()
    
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for step, (real_images, class_ids) in enumerate(progress_bar):
        optimizer.zero_grad()

        real_images = real_images.to(DEVICE, non_blocking=True)
        class_ids = class_ids.to(DEVICE, non_blocking=True)

        with amp_context:
            loss = sbgm.loss(real_images, class_ids)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update tqdm progress bar with loss
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    # --- Perform Sampling and Save Images (Intermediate Evaluation) ---
    if (epoch + 1) % SAMPLE_INTERVAL == 0 or epoch == EPOCHS - 1:
        print(f"\nSampling images using SDE at epoch {epoch + 1}...")
        classes_to_sample_list = list(range(min(NUM_CLASSES, 5)))
        generated_sample_images, _ = sbgm.sample(classes_to_sample_list, num_steps=NUM_STEPS, num_samples_per_cls=NUM_SAMPLES_PER_CLASS)
        # Save as a grid
        if generated_sample_images.nelement() > 0:  # Check if any images were generated
            grid = torchvision.utils.make_grid(generated_sample_images, nrow=NUM_SAMPLES_PER_CLASS)
            torchvision.utils.save_image(grid, f"sample_epoch_sde_{epoch + 1}.png")
            print(f"Saved sample images to sample_epoch_ode_{epoch + 1}.png")
        print("-" * 30)

        print(f"\nSampling images using PF-ODE at epoch {epoch + 1}...")
        classes_to_sample_list = list(range(min(NUM_CLASSES, 5)))
        generated_sample_images, _ = sbgm.sample_pf_ode(classes_to_sample_list, num_steps=NUM_STEPS, num_samples_per_cls=NUM_SAMPLES_PER_CLASS)
        # Save as a grid
        if generated_sample_images.nelement() > 0:  # Check if any images were generated
            grid = torchvision.utils.make_grid(generated_sample_images, nrow=NUM_SAMPLES_PER_CLASS)
            torchvision.utils.save_image(grid, f"sample_epoch_ode_{epoch + 1}.png")
            print(f"Saved sample images to sample_epoch_ode_{epoch + 1}.png")
        print("-" * 30)

    # Optional: Save model checkpoint
    if (epoch + 1) % CHECKPOINT_SAVE_INTERVAL == 0:
        torch.save(sbgm.score_model.state_dict(), f"dit_conditional_epoch_{epoch + 1}.pth")

print("Training finished.")