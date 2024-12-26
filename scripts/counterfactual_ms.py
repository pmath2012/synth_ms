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
from improved_diffusion.ms_datasets import load_data, LESION_LOAD_MAX, NUM_LESION_MAX

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
    
    out_dict = {
            "lesion_load": th.tensor(lesion_load, dtype=th.float32),
            "num_lesions": th.tensor(num_lesions, dtype=th.float32),
            "lesion": th.tensor(lesion, dtype=th.float32)  # Binary value
        }
    return out_dict

def preprocess_batch(batch, cond, df, microbatch=-1):
    """
    Preprocesses the batch and conditioning variables into a kwargs dictionary for microbatch processing.

    Args:
        batch (torch.Tensor): The input data batch.
        cond (dict): Dictionary of conditioning variables.
        microbatch (int): Size of the microbatch. If -1, the entire batch is processed.

    Returns:
        micro_batch (torch.Tensor): Processed batch tensor.
        model_kwargs (dict): Dictionary of model_kwargs with conditioning variables.
    """
    # Ensure the batch and cond are correctly shaped
    batch_size = batch.shape[0]

    # Handle case when microbatch is -1 (process the entire batch)
    if microbatch == -1:
        microbatch = batch_size

    # Initialize a dictionary to hold the processed model kwargs
    model_kwargs = {}
    cf_model_kwargs = {}

    cf_cond = get_random_lesion_info(df, batch_size)

    cond["mask"] = th.randn_like(cond.get("ground_truth"))  # Random mask for original condition
    cf_cond["mask"] = th.randn_like(cond.get("ground_truth"))  # Random mask for counterfactual condition


    # Iterate through the batch with a step size of `microbatch`
    for i in range(0, batch_size, microbatch):
        # Slice the batch for the current microbatch
        micro_batch = batch[i:i + microbatch].to(dist_util.dev())

        
        # Create a sliced cond dictionary for the microbatch
        micro_cond = {
            k: v[i:i + microbatch].to(dist_util.dev()) for k, v in cond.items()
        }

        micro_cf_cond = {
            k: v[i:i + microbatch].to(dist_util.dev()) for k, v in cf_cond.items()
        }

        # Store the sliced conditioning variables in model_kwargs
        model_kwargs.update(micro_cond)
        cf_model_kwargs.update(micro_cf_cond)

    load_changes = list(zip(micro_cond["lesion_load"], micro_cf_cond["lesion_load"]))
    num_lesion_changes = list(zip(micro_cond["num_lesions"], micro_cf_cond["num_lesions"]))
    lesion_changes = list(zip(micro_cond["lesion"], micro_cf_cond["lesion"]))

    # Return the processed microbatch and the kwargs dictionary
    return micro_batch, model_kwargs, cf_model_kwargs, (load_changes, num_lesion_changes, lesion_changes)

def gather_changes(changes):
    """
    Gathers a list of tuples across all GPUs. Each tuple contains two tensors.
    """
    gathered_originals = []
    gathered_modifieds = []
    
    for original, modified in changes:
        gathered_original = [th.zeros_like(original) for _ in range(dist.get_world_size())]
        gathered_modified = [th.zeros_like(modified) for _ in range(dist.get_world_size())]
        
        dist.all_gather(gathered_original, original)
        dist.all_gather(gathered_modified, modified)
        
        gathered_originals.append(th.cat(gathered_original, dim=0))
        gathered_modifieds.append(th.cat(gathered_modified, dim=0))
    
    return gathered_originals, gathered_modifieds


def main():
    args = create_argparser().parse_args()
    csv_path = os.path.join(args.root_dir, args.csv_file)
    df = pd.read_csv(csv_path)
    dist_util.setup_dist()
    logger.configure()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    logger.log(f"world size : {world_size}")
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
    logger.log("Creating data loader for medical images...")
    data = load_data(
        csv_file=args.csv_file,  
        root_dir=args.root_dir,  
        batch_size=args.batch_size, 
        distributed=world_size>1,
        train=False,
    )
    logger.log("sampling...")
    all_images = []
    all_labels = []
    all_orig = []
    all_gt = []
    all_dos = {"lesion_load":[], "num_lesions":[], "lesion":[]} 
    while len(all_images) * args.batch_size < args.num_samples:
        batch, cond = next(data)
        batch, model_kwargs, cf_model_kwargs, changes = preprocess_batch(batch, cond, df)
        _gt = model_kwargs.pop("ground_truth")
        #model_kwargs = {
        #    "mask" : th.randn(args.batch_size, 1, args.image_size, args.image_size),
        #}
        print(f"class conditioning : {args.class_cond}")
        if args.class_cond:
            classes = th.randint(
                low=0, high=2, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        
        # Abduct latent noise based on observed image
        logger.log("Performing Abduction...")
        abducted_noise = diffusion.ddim_sample_loop(
            model,
            (args.batch_size, 1, args.image_size, args.image_size),
            noise=batch,
            abduct=True,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        

        # Estimate counterfactual

        logger.log("Estimating counterfactual...")
        cf_sample = diffusion.ddim_sample_loop(
            model,
            (args.batch_size, 1, args.image_size, args.image_size),
            abduct=False,
            noise=abducted_noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=cf_model_kwargs
        )

        logger.log("Rescaling...")
        cf_sample = (cf_sample * 255).clamp(0, 255).to(th.uint8)
        cf_sample = cf_sample.permute(0, 2, 3, 1)
        cf_sample = cf_sample.contiguous()

        orig_img = (batch * 255).clamp(0, 255).to(th.uint8)
        orig_img = orig_img.permute(0, 2, 3, 1)
        orig_img = orig_img.contiguous()

        gt_mask = (_gt * 255).clamp(0, 255).to(th.uint8)
        gt_mask = gt_mask.permute(0, 2, 3, 1)
        gt_mask = gt_mask.contiguous()

        logger.log("Gathering samples from other GPUS -> CF...")
        gathered_samples = [th.zeros_like(cf_sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, cf_sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        
        logger.log("Gathering samples from other GPUS -> Orig...")
        gathered_orig = [th.zeros_like(orig_img) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_orig, orig_img)  # gather not supported with NCCL
        all_orig.extend([sample.cpu().numpy() for sample in gathered_orig])
        
        logger.log("Gathering samples from other GPUS -> GT...")
        gathered_gt = [th.zeros_like(gt_mask) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_gt, gt_mask)  # gather not supported with NCCL
        all_gt.extend([sample.cpu().numpy() for sample in gathered_gt])
        logger.log("Gathering changes from all GPUs...")
        


        #### does not work
        logger.log("Gathering lesion load...")
        gathered_lesion_load = gather_changes(changes[0])
        logger.log("Gathering num_lesions...")
        gathered_num_lesions = gather_changes(changes[1])
        logger.log("Gathering lesions...")
        gathered_lesions = gather(changes[2])

        all_dos["lesion_load"].append(gathered_lesion_load)
        all_dos["num_lesions"].append(gathered_num_lesions)
        all_dos["lesion"].append(gathered_lesions) 

        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    logger.log("Concatenating data now...")
    print(f"Here for device {dist.get_rank()}")
    cfs = np.concatenate(all_images, axis=0)
    cfs = cfs[: args.num_samples]
    img = np.concatenate(all_orig, axis=0)
    img = img[: args.num_samples]
    msk = np.concatenate(all_gt, axis=0)
    msk = msk[: args.num_samples]
        
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in cfs.shape])
        if args.output_dir is None:
            out_path = os.path.join(logger.get_dir(), f"cf_samples_{shape_str}.npz")
        else:
            out_path = os.path.join(args.output_dir, f"cf_samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, cfs, label_arr)
        else:
            np.savez(out_path, cfs, img, msk, all_dos)

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
        csv_file="",
        root_dir="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
