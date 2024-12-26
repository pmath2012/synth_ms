"""
Train a conditional diffusion model on medical images with masks and lesion information on an HPC environment.
"""
import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from improved_diffusion import dist_util, logger
from improved_diffusion.ms_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    create_conditional_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from glasses.models.segmentation.unet import UNet

# Clear distributed environment variables if running in non-distributed mode
#os.environ.pop('RANK', None)
#os.environ.pop('WORLD_SIZE', 1)
#os.environ.pop('LOCAL_RANK', None)

def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=256,
        num_channels=64,
        num_res_blocks=2,
        num_heads=2,
        num_heads_upsample=-1,
        attention_resolutions="16",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )

def load_segmentation_model(checkpoint_path, device):
    """
    Loads the segmentation model from the given checkpoint path, freezes its weights,
    and moves it to the correct device (GPU or CPU).
    """
    # Define your model architecture (you need to define your model class elsewhere)
    segmentation_model = UNet(n_classes=1, in_channels=1)  # Replace with your actual model class

    # Load the model weights from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=True)  # Load on CPU first
    segmentation_model.load_state_dict(checkpoint['model_state_dict'])

    # Freeze the model's parameters (no gradient updates)
    for param in segmentation_model.parameters():
        param.requires_grad = False

    # Set the model to evaluation mode
    segmentation_model.eval()

    # Move the model to the correct device (GPU if available)
    segmentation_model.to(device)

    return segmentation_model

def main():
    # Parse arguments
    args = create_argparser().parse_args()

    run_training(args)

def run_training(args):
    # Set up distributed environment (DDP)
    dist_util.setup_dist()
    # Set device based on distributed setup
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    logger.configure()
    logger.log(f"World Size : {world_size}")
    logger.log("Creating conditional model and diffusion process...")
    device = dist_util.dev()
    segmentation_model = load_segmentation_model(args.segmentation_model, device=device)

    model, diffusion = create_conditional_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        segmentation_model=segmentation_model,
    )
    model.to(device)

    # Wrap the model in DDP only if distributed
    if world_size>1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist_util.dev()], output_device=dist_util.dev()
        )
    # Schedule sampler
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)


    logger.log("Creating data loader for medical images...")
    data = load_data(
        csv_file=args.csv_file,  
        root_dir=args.root_dir,  
        batch_size=args.batch_size,  
        distributed=world_size>1,  # Enable distributed data loading
    )

    # Start training loop
    logger.log("Starting training loop...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

    if world_size>1:
        dist.destroy_process_group()

def create_argparser():
    defaults = dict(
        root_dir="",
        csv_file="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        world_size=1,
        class_cond=True,
        rescale_learned_sigmas=False,
        learn_sigma=False,
        segmentation_model="",  # <-- Added segmentation model argument
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
