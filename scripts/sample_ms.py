"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import pandas as pd
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    create_conditional_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from train_ms import model_and_diffusion_defaults, load_segmentation_model
from improved_diffusion.ms_datasets import LESION_LOAD_MAX, NUM_LESION_MAX


def get_random_lesion_info(df, batch_size):
    """
    Loads the CSV and selects batch_size random rows containing the lesion information.
    Applies the same normalization as during training.
    Returns the lesion_load, num_lesions, and lesion (health status) as tensors for the batch.
    """
    
    # Randomly select batch_size rows from the dataframe
    random_rows = df.sample(n=batch_size)
    
    # Extract and normalize the relevant columns for all selected rows
    num_lesions = random_rows['num_lesions'].astype(float).values / NUM_LESION_MAX  # Scale based on max value during training
    lesion_load = random_rows['lesion_load'].astype(float).values / LESION_LOAD_MAX  # Scale based on max value during training
    lesion = random_rows['lesion'].astype(int).values  # No scaling needed for binary values
    
    # Convert to tensors
    num_lesions = th.tensor(num_lesions).float().unsqueeze(1)  # Reshape to (batch_size, 1)
    lesion_load = th.tensor(lesion_load).float().unsqueeze(1)  # Reshape to (batch_size, 1)
    lesion = th.tensor(lesion).float().unsqueeze(1)            # Reshape to (batch_size, 1)
    
    return lesion_load, num_lesions, lesion

def main():
    args = create_argparser().parse_args()
    df = pd.read_csv(args.csv_file)
    dist_util.setup_dist()
    logger.configure()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    logger.log("creating model and diffusion...")
    device = dist_util.dev()
    segmentation_model = load_segmentation_model(args.segmentation_model, device=device)

    model, diffusion = create_conditional_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        segmentation_model=segmentation_model,
    )
    model.to(device)
    if world_size>1:
        model = th.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist_util.dev()], output_device=dist_util.dev()
        )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        lesion_load, num_lesions, lesion = get_random_lesion_info(df, args.batch_size)
        model_kwargs = {
            "mask" : th.randn(args.batch_size, 1, args.image_size, args.image_size),
            "lesion_load": lesion_load,
            "num_lesions":num_lesions,
            "lesion":lesion,
        }
        print(f"class conditioning : {args.class_cond}")
        if args.class_cond:
            classes = th.randint(
                low=0, high=2, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 1, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = (sample * 255).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        if args.output_dir is None:
            out_path = os.path.join(logger.get_dir(), f"test_samples_{shape_str}.npz")
        else:
            out_path = os.path.join(args.output_dir, f"test_samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=64,
        batch_size=8,
        use_ddim=False,
        model_path="",
        output_dir=None,
        segmentation_model=None,
        csv_file=None
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()


