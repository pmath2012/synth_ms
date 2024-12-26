import os
import argparse
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2

from improved_diffusion.script_util import (
    create_conditional_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from train_ms import model_and_diffusion_defaults, load_segmentation_model
from improved_diffusion.ms_datasets import load_data, LESION_LOAD_MAX, NUM_LESION_MAX

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic images using a diffusion model.")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory for image masks and outputs')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file containing image metadata')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for image generation')
    parser.add_argument('--checkpoint', type=str, required=True )
    return parser.parse_args()

def return_args(root_dir, filename, lesion_load, num_lesions, lesion):
    """
    Prepare model_kwargs including mask, lesion statistics, and the image name.
    """
    mask_path = os.path.join(root_dir, filename)
    mask = Image.open(mask_path).convert('L')
    mask = transforms.functional.to_tensor(mask)  # Convert to tensor (values between 0 and 1)
    mask = torch.where(mask > 0.5, torch.tensor(1.0), torch.tensor(0.0))  # Binarize the mask
    
    image_name = filename.replace("label", "image").replace("mask", "image")
    
    return {
        "mask": mask,  # Add batch and channel dimensions
        "lesion_load": torch.tensor(lesion_load, dtype=torch.float32),
        "num_lesions": torch.tensor(num_lesions, dtype=torch.float32),
        "lesion": torch.tensor(lesion, dtype=torch.float32),
        "image_name": image_name
    }

def preprocess(model_kwargs, device):
    """
    Move model_kwargs to the specified device.
    """
    return {k: v.to(device) for k, v in model_kwargs.items()}

def create_image(sample_fn, model, df, root_dir, device, batch_size):
    """
    Generate images in batches using the diffusion model and save them.
    
    Args:
        sample_fn: The function used to sample from the diffusion model.
        model: The trained diffusion model.
        df: DataFrame containing metadata (label, lesion_load, num_lesions, lesion).
        root_dir: Directory containing the mask files.
        device: Torch device (GPU).
        batch_size: Number of images to process in one batch.
    """
    for batch_idx in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[batch_idx:batch_idx + batch_size]
        model_kwargs_list = []
        image_names = []

        for _, row in batch_df.iterrows():
            model_kwargs = return_args(
                root_dir=root_dir,
                filename=row['label'],  # Assumes 'label' column contains the mask filename
                lesion_load=row['lesion_load'] / LESION_LOAD_MAX**2,
                num_lesions=row['num_lesions'] / NUM_LESION_MAX**2,
                lesion=row['lesion']
            )
            image_name = model_kwargs.pop("image_name")
            model_kwargs_list.append(preprocess(model_kwargs, device))
            image_names.append(image_name)
        
        # Stack model_kwargs for batch processing
        masks = torch.stack([kwargs["mask"] for kwargs in model_kwargs_list]).to(device)  # Shape: (batch_size, 1, 256, 256)
        lesion_loads = torch.stack([kwargs["lesion_load"] for kwargs in model_kwargs_list]).to(device)
        num_lesions = torch.stack([kwargs["num_lesions"] for kwargs in model_kwargs_list]).to(device)
        lesions = torch.stack([kwargs["lesion"] for kwargs in model_kwargs_list]).to(device)
        
        # Create the final batch model_kwargs
        batch_model_kwargs = {
            "mask": masks,
            "lesion_load": lesion_loads,
            "num_lesions": num_lesions,
            "lesion": lesions
        }

        # Generate the batch of samples using the diffusion model
        with torch.no_grad():
            samples = sample_fn(
                model,
                (batch_size, 1, 256, 256),  # Batch size and image size
                clip_denoised=True,
                model_kwargs=batch_model_kwargs,
            )
        
        # Post-process and save each sample in the batch
        for i in range(samples.size(0)):  # Iterate over the batch
            sample = samples[i]
            sample = (sample * 255).clamp(0, 255).to(torch.uint8)
            sample = sample.permute(1, 2, 0).squeeze().cpu().numpy()  # Shape: (256, 256)
            
            # Save the generated image
            save_path = os.path.join(root_dir, image_names[i])
            cv2.imwrite(save_path, sample)

            print(f"Saved image {image_names[i]} to {save_path}")

def execute(args,device):

    # Load the CSV file
    df = pd.read_csv(os.path.join(args.root_dir, args.csv_file))
    
    model, diffusion = create_conditional_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )


    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    new_state_dict = {}
    for key, value in checkpoint.items():
        # Remove 'module.' from the keys if it exists
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' from the key
        else:
            new_key = key
        new_state_dict[new_key] = value

    model.to(device)
    model.load_state_dict(new_state_dict)
    model.eval()
    
    sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
    # Call the image generation function
    create_image(sample_fn, model, df, args.root_dir, device, args.batch_size)
    

def main():
    args = parse_args()
    defaults = {}
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        checkpoint="/path/to/saved/model/sm.pt",
        output_dir=None,
        segmentation_model=None,
        learn_sigma=False,
        use_fp16=False,
        csv_file="test.csv",
        root_dir="/path/to/root/data/",
        image_size=256,
        rescale_learned_sigmas=False,
        class_cond=False,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults["rescale_learned_sigmas"] = False
    args_dict = vars(args)
    defaults.update(args_dict)
    final_args = argparse.Namespace(**defaults)
    print(final_args)
    
    # Set up device (assumes one GPU is visible through CUDA_VISIBLE_DEVICES)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    execute(final_args, device)

if __name__ == "__main__":
    main()
