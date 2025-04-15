import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from torchvision.utils import save_image



class CycleGan():

    def __init__(self, 
                 disc_hf, 
                 disc_lf, 
                 gen_hf, 
                 gen_lf, 
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.disc_hf = disc_hf.to(device)
        self.disc_lf = disc_lf.to(device)
        self.gen_hf = gen_hf.to(device)
        self.gen_lf = gen_lf.to(device)
        self.device = device
        
    
    def train(self, train_loader, disc_optimizer, gen_optimizer, l1, mse, d_scaler, g_scaler, LAMBDA_CYCLE):
        H_reals = 0
        H_fakes = 0
        self.gen_hf.train()
        self.gen_lf.train()
        self.disc_hf.train()
        self.disc_lf.train()

        loop = tqdm(train_loader, leave=True)

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


            
    

