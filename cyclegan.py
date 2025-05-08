import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from torchvision.utils import save_image
import os
from gans import Discriminator, Generator
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from datahandlers import DataTransform_M4RAW, LF_M4RawDataset, HF_MRI_Dataset, UnpairedMergedDataset, lf_hf_collate_fn
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


###### Transforms #######
transform_hf = A.Compose([
    A.Resize(256, 256),
    ToTensorV2()
])
lf_transform = DataTransform_M4RAW(img_size=256, combine_coil=True)

#### Low Field Datasets: Train, Test, Val ####
lf_dataset_train = LF_M4RawDataset(root_dir='dataset/low_field/multicoil_train', transform=lf_transform)
lf_dataset_val = LF_M4RawDataset(root_dir="dataset/low_field/multicoil_val", transform=lf_transform)
#lf_dataset_test = LF_M4RawDataset(root_dir="dataset/low_field/multicoil_test", transform=lf_transform)

#### High Field Datasets: Train, Test, Val
hf_dataset_train = HF_MRI_Dataset(root_dir="dataset/brain_fastMRI_DICOM/fastMRI_brain_DICOM", 
                                  transform=transform_hf,
                                  split="train")
hf_dataset_val = HF_MRI_Dataset(root_dir="dataset/brain_fastMRI_DICOM/fastMRI_brain_DICOM", 
                                  transform=transform_hf,
                                  split="val")
#hf_dataset_test = HF_MRI_Dataset(root_dir="dataset/brain_fastMRI_DICOM/fastMRI_brain_DICOM", 
#                                  transform=transform_hf,
#                                  split="test")


#### Concat Datasets ####
train_set = UnpairedMergedDataset(lf_dataset_train, hf_dataset_train)
val_set = UnpairedMergedDataset(lf_dataset_val, hf_dataset_val)
#test_set = UnpairedMergedDataset(lf_dataset_test, hf_dataset_test)

#### DataLoaders ####
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=6, collate_fn=lf_hf_collate_fn)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=2, shuffle=False, num_workers=4, collate_fn=lf_hf_collate_fn)
#test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4, collate_fn=lf_hf_collate_fn)


def warmup(
    loader,
    disc_hf, disc_lf,
    gen_hf, gen_lf,
    opt_disc,
    mse,
    d_scaler,
    device,
    warmup_steps=100
):
    """
    Pre-train the discriminators for `warmup_steps` batches before starting generator updates.
    """
    it = iter(loader)
    for i in range(warmup_steps):
        try:
            data = next(it)
        except StopIteration:
            break
        low_field, high_field = data[0].to(device), data[1].to(device)
        with torch.cuda.amp.autocast():
            # generate fake without updating generators
            fake_h = gen_hf(low_field).detach()
            fake_l = gen_lf(high_field).detach()
            # HF discriminator loss
            D_H_real = disc_hf(high_field)
            D_H_fake = disc_hf(fake_h)
            D_H_loss = mse(D_H_real, torch.ones_like(D_H_real)) + \
                       mse(D_H_fake, torch.zeros_like(D_H_fake))
            # LF discriminator loss
            D_l_real = disc_lf(low_field)
            D_l_fake = disc_lf(fake_l)
            D_l_loss = mse(D_l_real, torch.ones_like(D_l_real)) + \
                       mse(D_l_fake, torch.zeros_like(D_l_fake))
            D_loss = 0.5 * (D_H_loss + D_l_loss)
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()


class CycleGan():

    def __init__(self, 
                 disc_hf, 
                 disc_lf, 
                 gen_hf, 
                 gen_lf, 
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        wandb.init()
        self.disc_hf = disc_hf.to(device)
        self.disc_lf = disc_lf.to(device)
        self.gen_hf = gen_hf.to(device)
        self.gen_lf = gen_lf.to(device)
        self.device = device
        self.best_loss = float("inf")
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("val_outputs", exist_ok=True)
        
    
    def train(self, train_loader, disc_optimizer, gen_optimizer, l1, mse, d_scaler, g_scaler, LAMBDA_CYCLE, warmup_steps=100):

        # 1) Warm-up discriminators
        if warmup_steps != 0 :
            warmup(
                train_loader,
                self.disc_hf,
                self.disc_lf,
                self.gen_hf,
                self.gen_lf,
                disc_optimizer,
                mse,
                d_scaler,
                self.device,
                warmup_steps
            )

        H_reals = 0
        H_fakes = 0
        self.gen_hf.train()
        self.gen_lf.train()
        self.disc_hf.train()
        self.disc_lf.train()

        loop = tqdm(train_loader, leave=True, desc="Training")

        for idx, data in enumerate(loop):
            low_field = data[0].to(self.device)
            high_field = data[1].to(self.device)

            with torch.amp.autocast("cuda"):
                fake_h = self.gen_hf(low_field)
                D_H_real = self.disc_hf(high_field)
                D_H_fake = self.disc_hf(fake_h.detach())
                H_reals += D_H_real.mean().item()
                H_fakes += D_H_fake.mean().item()
                D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
                D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
                D_H_loss = D_H_real_loss + D_H_fake_loss

                fake_l = self.gen_lf(high_field)
                D_l_real = self.disc_lf(low_field)
                D_l_fake = self.disc_lf(fake_l.detach())
                D_l_real_loss = mse(D_l_real, torch.ones_like(D_l_real))
                D_l_fake_loss = mse(D_l_fake, torch.zeros_like(D_l_fake))
                D_l_loss = D_l_real_loss + D_l_fake_loss

                D_loss = (D_H_loss + D_l_loss) / 2
            
            disc_optimizer.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(disc_optimizer)
            d_scaler.update()


            with torch.amp.autocast("cuda"):
                D_H_fake = self.disc_hf(fake_h)
                D_l_fake = self.disc_lf(fake_l)
                loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
                loss_G_Z = mse(D_l_fake, torch.ones_like(D_l_fake))

                # cycle losses
                cycle_l = self.gen_lf(fake_h)
                cycle_h = self.gen_hf(fake_l)
                cycle_l_loss = l1(low_field, cycle_l)
                cycle_h_loss = l1(high_field, cycle_h)
                # identity losses
                identity_l = self.gen_lf(low_field)
                identity_h = self.gen_hf(high_field)
                identity_l_loss = l1(low_field, identity_l)
                identity_h_loss = l1(high_field, identity_h)
                G_loss = (
                    loss_G_Z
                    + loss_G_H
                    + cycle_l_loss * LAMBDA_CYCLE
                    + cycle_h_loss * LAMBDA_CYCLE
                    +identity_h_loss * 0.5 * LAMBDA_CYCLE
                    + identity_l_loss * 0.5 * LAMBDA_CYCLE
        
                )

            gen_optimizer.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(gen_optimizer)
            g_scaler.update()
            if idx % 500 == 0:
                img_lh = fake_h[0].detach().cpu().numpy() * 0.5 + 0.5

                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(img_lh, cmap="gray")
                axs[0].set_title("High-res")
                axs[0].axis("off")

                axs[1].imshow(low_field[0].detach().cpu().numpy()*0.5+0.5, cmap="gray")
                axs[1].set_title("Low-res")
                axs[1].axis("off")
                plt.suptitle(f"Sample {idx}")
                plt.tight_layout()
                plt.savefig(f"outputs/h_l{idx}.png", bbox_inches='tight')
                plt.close()




            if idx % 20 == 0:
                wandb.log({
                    "D_loss": D_loss.item(),
                    "G_loss": G_loss.item(),
                    "Cycle_Loss_LF": cycle_l_loss.item(),
                    "Cycle_Loss_HF": cycle_h_loss.item(),
                    "Identity_Loss_LF": identity_l_loss.item(),
                    "Identity_Loss_HF": identity_h_loss.item(),
                    "D_H_real": D_H_real.mean().item(),
                    "D_H_fake": D_H_fake.mean().item()
                })

            loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


            


    @torch.no_grad()
    def validate(self, val_loader, l1, mse, LAMBDA_CYCLE, step=None, log_images=False):
        self.gen_hf.eval()
        self.gen_lf.eval()
        H_reals = 0
        H_fakes = 0
        val_loop = tqdm(val_loader, leave=True, desc="Validation")

        for idx, data in enumerate(val_loop):
            low_field = data[0].to(self.device)
            high_field = data[1].to(self.device)
            
            with torch.amp.autocast("cuda"):
                fake_h = self.gen_hf(low_field)
                D_H_real = self.disc_hf(high_field)
                D_H_fake = self.disc_hf(fake_h.detach())
                H_reals += D_H_real.mean().item()
                H_fakes += D_H_fake.mean().item()
                D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
                D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
                D_H_loss = D_H_real_loss + D_H_fake_loss

                fake_l = self.gen_lf(high_field)
                D_l_real = self.disc_lf(low_field)
                D_l_fake = self.disc_lf(fake_l.detach())
                D_l_real_loss = mse(D_l_real, torch.ones_like(D_l_real))
                D_l_fake_loss = mse(D_l_fake, torch.zeros_like(D_l_fake))
                D_l_loss = D_l_real_loss + D_l_fake_loss

                D_loss = (D_H_loss + D_l_loss) / 2

            with torch.amp.autocast("cuda"):
                D_H_fake = self.disc_hf(fake_h)
                D_l_fake = self.disc_lf(fake_l)
                loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
                loss_G_Z = mse(D_l_fake, torch.ones_like(D_l_fake))

                # cycle losses
                cycle_l = self.gen_lf(fake_h)
                cycle_h = self.gen_hf(fake_l)
                cycle_l_loss = l1(low_field, cycle_l)
                cycle_h_loss = l1(high_field, cycle_h)
                # identity losses
                identity_l = self.gen_lf(low_field)
                identity_h = self.gen_hf(high_field)
                identity_l_loss = l1(low_field, identity_l)
                identity_h_loss = l1(high_field, identity_h)
                G_loss = (
                    loss_G_Z
                    + loss_G_H
                    + cycle_l_loss * LAMBDA_CYCLE
                    + cycle_h_loss * LAMBDA_CYCLE
                    +identity_h_loss * 0.5 * LAMBDA_CYCLE
                    + identity_l_loss * 0.5 * LAMBDA_CYCLE
        
                )


            # Log and save images for the first batch
            if log_images and idx % 200 == 0:
                os.makedirs("val_outputs", exist_ok=True)
                img_lh = fake_h[0].detach().cpu().numpy() * 0.5 + 0.5

                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(img_lh, cmap="gray")
                axs[0].set_title("High-res")
                axs[0].axis("off")

                axs[1].imshow(low_field[0].detach().cpu().numpy()*0.5+0.5, cmap="gray")
                axs[1].set_title("Low-res")
                axs[1].axis("off")
                plt.suptitle(f"Sample {idx}")
                plt.tight_layout()
                plt.savefig(f"val_outputs/h_l{idx}.png", bbox_inches='tight')
                plt.close()


            if idx % 20 == 0:
                wandb.log({
                    "D_loss": D_loss.item(),
                    "G_loss": G_loss.item(),
                    "Cycle_Loss_LF": cycle_l_loss.item(),
                    "Cycle_Loss_HF": cycle_h_loss.item(),
                    "Identity_Loss_LF": identity_l_loss.item(),
                    "Identity_Loss_HF": identity_h_loss.item(),
                    "D_H_real": D_H_real.mean().item(),
                    "D_H_fake": D_H_fake.mean().item()
                })

        self.save_best_model(cycle_l_loss, step=step)

        self.gen_hf.train()
        self.gen_lf.train()
    
    def save_best_model(self, mean_cycle_loss, step=None):
        if mean_cycle_loss <= self.best_loss:
            self.best_loss = mean_cycle_loss
            os.makedirs("checkpoints", exist_ok=True)

            torch.save(self.gen_hf.state_dict(), "checkpoints/best_gen_hf.pth")
            torch.save(self.gen_lf.state_dict(), "checkpoints/best_gen_lf.pth")
            torch.save(self.disc_hf.state_dict(), "checkpoints/best_disc_hf.pth")
            torch.save(self.disc_lf.state_dict(), "checkpoints/best_disc_lf.pth")

            if step is not None:
                wandb.log({"best_model_saved": True}, step=step)
            print(f"Saved new best model with cycle loss: {mean_cycle_loss:.4f}")

    def save_model(self):
        os.makedirs("checkpoints/frequent", exist_ok=True)
        torch.save(self.gen_hf.state_dict(), "checkpoints/frequent/last_gen_hf.pth")
        torch.save(self.gen_lf.state_dict(), "checkpoints/frequent/last_gen_lf.pth")
        torch.save(self.disc_hf.state_dict(), "checkpoints/frequent/last_disc_hf.pth")
        torch.save(self.disc_lf.state_dict(), "checkpoints/frequent/last_disc_lf.pth")
    
    def load_model_from_ckpts(self):
        self.gen_hf.load_state_dict(torch.load("checkpoints/best_gen_hf.pth"))
        self.gen_lf.load_state_dict(torch.load("checkpoints/best_gen_lf.pth"))
        self.disc_hf.load_state_dict(torch.load("checkpoints/best_disc_hf.pth"))
        self.disc_lf.load_state_dict(torch.load("checkpoints/best_disc_lf.pth"))




def main():
    disc_H = Discriminator(in_channels=1)
    disc_Z = Discriminator(in_channels=1)
    gen_Z = Generator(img_channels=1, num_residuals=9)
    gen_H = Generator(img_channels=1, num_residuals=9)

    # use Adam Optimizer for both generator and discriminator
    opt_disc = torch.optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=2e-4,
        betas=(0.5, 0.999),
    )

    opt_gen = torch.optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=2e-4,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    g_scaler = torch.amp.GradScaler("cuda")
    d_scaler = torch.amp.GradScaler("cuda")

    model = CycleGan(disc_H, disc_Z, gen_H, gen_Z)

    for epoch in range(200):
        print(f"Epoch : {epoch} Training")
        model.train(train_loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, 10, 0)
        print(f"Epoch : {epoch} Validation")
        model.validate(val_loader, L1, mse, 10, step=epoch, log_images=True)
        if epoch % 50 == 0 and epoch !=0:
            model.save_model()

        
        
    model.save_model()

