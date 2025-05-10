import torch
from tqdm import tqdm
from torchvision.utils import save_image
import os
from gans import  Generator
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datahandlers import DataTransform_M4RAW, LF_M4RawDataset, HF_MRI_Dataset, UnpairedMergedDataset, lf_hf_collate_fn
import torchvision.utils as vutils
device="cuda"


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
#hf_dataset_test = HF_MRI_Dataset(root_dir="dataset/fastMRI_brain_DICOM/fastMRI_brain_DICOM", 
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

gen_h = Generator(img_channels=1,num_residuals=9).to(device)
gen_h.load_state_dict(torch.load("checkpoints/best_hf.pth"))

gen_l=Generator(img_channels=1, num_residuals=9).to(device)
gen_l.load_state_dict(torch.load("checkpoints/best_gen_lf.pth"))

print("loaded")

@torch.no_grad()
def validate(gen_hf, gen_lf, val_loader, l1, LAMBDA_CYCLE, step=None, log_images=True, save_dir="val_images"):
    gen_hf.eval()
    gen_lf.eval()

    total_cycle_l_loss = 0.0
    total_cycle_h_loss = 0.0
    num_batches = 0

    os.makedirs(save_dir, exist_ok=True)

    val_loop = tqdm(val_loader, leave=True, desc="Validation")

    for idx, data in enumerate(val_loop):
        low_field = data[0].to(device)
        high_field = data[1].to(device)

        with torch.amp.autocast("cuda"):
            fake_h = gen_hf(low_field)
            fake_l = gen_lf(high_field)

            cycle_l = gen_lf(fake_h)
            cycle_h = gen_hf(fake_l)

            cycle_l_loss = l1(low_field, cycle_l)
            cycle_h_loss = l1(high_field, cycle_h)


        total_cycle_l_loss += cycle_l_loss.item()
        total_cycle_h_loss += cycle_h_loss.item()
        num_batches += 1

        if log_images and idx % 10 == 0:  # save every 5th batch
            for i in range(low_field.size(0)):
                lf_img = low_field[i].detach().cpu()*0.5+0.5
                fake_h_img = fake_h[i].detach().cpu()*0.5+0.5

                concat = torch.cat([lf_img, fake_h_img], dim=-1)
                vutils.save_image(
                    concat,
                    os.path.join(save_dir, f"val_step{step}_idx{idx}_img{i}.png"),
                    normalize=True,
                )

    avg_cycle_l_loss = total_cycle_l_loss / num_batches
    avg_cycle_h_loss = total_cycle_h_loss / num_batches

    print(f"\nFinal Cycle L Loss (LF→HF→LF): {avg_cycle_l_loss:.4f}")
    print(f"Final Cycle H Loss (HF→LF→HF): {avg_cycle_h_loss:.4f}")
    print(f"Total G Loss: {(avg_cycle_l_loss + avg_cycle_h_loss) * LAMBDA_CYCLE:.4f}")


def main():
    validate(gen_hf=gen_h,
            gen_lf=gen_l,
            val_loader=val_loader,
            l1= torch.nn.L1Loss(),
            LAMBDA_CYCLE=10,
            log_images=True,
            save_dir="val_images"
            )

main()



