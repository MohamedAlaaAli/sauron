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
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("val_outputs", exist_ok=True)
        
    
    def train(self, train_loader, disc_optimizer, gen_optimizer, l1, mse, d_scaler, g_scaler, LAMBDA_CYCLE):
        H_reals = 0
        H_fakes = 0
        self.gen_hf.train()
        self.gen_lf.train()
        self.disc_hf.train()
        self.disc_lf.train()

        loop = tqdm(train_loader, leave=True, desc="Training")

        for idx, (low_field, high_field) in enumerate(loop):
            low_field = low_field.to(self.device)
            high_field = high_field.to(self.device)

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
            
                G_loss = (
                    loss_G_Z
                    + loss_G_H
                    + cycle_l_loss * LAMBDA_CYCLE
                    + cycle_h_loss * LAMBDA_CYCLE
        
                )

            gen_optimizer.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(gen_optimizer)
            g_scaler.update()
            if idx % 200 == 0:
                save_image(fake_h.detach() * 0.5 + 0.5, f"outputs/hf_{idx}.png")
                save_image(fake_l.detach() * 0.5 + 0.5, f"outputs/lf{idx}.png")

            wandb.log({
                "D_loss": D_loss.item(),
                "G_loss": G_loss.item(),
                "Cycle_Loss_LF": cycle_l_loss.item(),
                "Cycle_Loss_HF": cycle_h_loss.item(),
                "D_H_real": D_H_real.mean().item(),
                "D_H_fake": D_H_fake.mean().item()
            })

            loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


            


    @torch.no_grad()
    def validate(self, val_loader, l1, LAMBDA_CYCLE, step=None, log_images=False):
        self.gen_hf.eval()
        self.gen_lf.eval()
        
        total_cycle_l_loss = 0
        total_cycle_h_loss = 0

        val_loop = tqdm(val_loader, leave=True, desc="Validation")

        for idx, (low_field, high_field) in enumerate(val_loop):
            low_field = low_field.to(self.device)
            high_field = high_field.to(self.device)

            fake_h = self.gen_hf(low_field)
            fake_l = self.gen_lf(high_field)

            cycle_l = self.gen_lf(fake_h)
            cycle_h = self.gen_hf(fake_l)

            cycle_l_loss = l1(low_field, cycle_l)
            cycle_h_loss = l1(high_field, cycle_h)

            total_cycle_l_loss += cycle_l_loss.item()
            total_cycle_h_loss += cycle_h_loss.item()

            val_loop.set_postfix({
                "Cycle_LF": cycle_l_loss.item(),
                "Cycle_HF": cycle_h_loss.item()
            })

            # Log and save images for the first batch
            if log_images and idx == 0:
                os.makedirs("val_outputs", exist_ok=True)
                save_image(fake_h * 0.5 + 0.5, f"val_outputs/fake_h_step{step}.png")
                save_image(fake_l * 0.5 + 0.5, f"val_outputs/fake_l_step{step}.png")

                wandb.log({
                    "val/fake_h": wandb.Image(fake_h[0].detach().cpu()),
                    "val/fake_l": wandb.Image(fake_l[0].detach().cpu())
                }, step=step)

        mean_cycle_l = total_cycle_l_loss / len(val_loader)
        mean_cycle_h = total_cycle_h_loss / len(val_loader)

        wandb.log({
            "val/Cycle_Loss_LF": mean_cycle_l,
            "val/Cycle_Loss_HF": mean_cycle_h,
            "val/Total_Cycle_Loss": (mean_cycle_l + mean_cycle_h) * LAMBDA_CYCLE
        }, step=step)

        total_cycle_loss = (mean_cycle_l + mean_cycle_h) * LAMBDA_CYCLE
        self.save_best_model(total_cycle_loss, step=step)
        
        self.gen_hf.train()
        self.gen_lf.train()
    
    def save_best_model(self, mean_cycle_loss, step=None):
        if mean_cycle_loss < self.best_loss:
            self.best_loss = mean_cycle_loss
            os.makedirs("checkpoints", exist_ok=True)

            torch.save(self.gen_hf.state_dict(), "checkpoints/best_gen_hf.pth")
            torch.save(self.gen_lf.state_dict(), "checkpoints/best_gen_lf.pth")
            torch.save(self.disc_hf.state_dict(), "checkpoints/best_disc_hf.pth")
            torch.save(self.disc_lf.state_dict(), "checkpoints/best_disc_lf.pth")

            if step is not None:
                wandb.log({"best_model_saved": True}, step=step)
            print(f"Saved new best model with cycle loss: {mean_cycle_loss:.4f}")




class HorseZebraDataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform=None):
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform

        self.zebra_images = os.listdir(root_zebra)
        self.horse_images = os.listdir(root_horse)
        self.length_dataset = max(len(self.zebra_images), len(self.horse_images))
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        zebra_img = self.zebra_images[index % self.zebra_len]
        horse_img = self.horse_images[index % self.horse_len]

        zebra_path = os.path.join(self.root_zebra, zebra_img)
        horse_path = os.path.join(self.root_horse, horse_img)

        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augmentations["image"]
            horse_img = augmentations["image0"]

        return zebra_img, horse_img



def main():
    disc_H = Discriminator(in_channels=3)
    disc_Z = Discriminator(in_channels=3)
    gen_Z = Generator(img_channels=3, num_residuals=9)
    gen_H = Generator(img_channels=3, num_residuals=9)

    # use Adam Optimizer for both generator and discriminator
    opt_disc = torch.optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=1e-5,
        betas=(0.5, 0.999),
    )

    opt_gen = torch.optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=1e-5,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
    )


    dataset = HorseZebraDataset(
        root_horse="my_horse2zebra/train" + "/train_A",
        root_zebra="my_horse2zebra/train" + "/train_B",
        transform=transforms,
    )
    val_dataset = HorseZebraDataset(
        root_horse="my_horse2zebra/test" + "/test_A",
        root_zebra="my_horse2zebra/test" + "/test_B",
        transform=transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    g_scaler = torch.amp.GradScaler("cuda")
    d_scaler = torch.amp.GradScaler("cuda")

    model = CycleGan(disc_H, disc_Z, gen_H, gen_Z)

    for epoch in range(50):
        model.train(loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, 10)
        model.validate(val_loader, L1, 10, step=epoch, log_images=(epoch % 5 == 0))


main()