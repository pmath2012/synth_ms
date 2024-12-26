# Synth-MS : Synthetic dataset for MS Lesion Segmentation

This is the codebase is based on the code for [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672).

# Usage

Please refer to the original improved-diffusion codebase for installation instructions. Additional dependencies include the [monai](https://github.com/Project-MONAI/tutorials) codebase and the [glasses](https://github.com/FrancescoSaverioZuppichini/glasses/tree/master/glasses/) codebase

## Training auxiliary segmentation model
We utilise a trained UNet architecture to train our proposed lesion guided diffusion model
use the following in the basic_unet folder to train the unet:
```
# Run the Python script in the background
nohup python train_unet.py \
  --train_csv $TRAIN_CSV \
  --val_csv $VAL_CSV \
  --root_dir $ROOT_DIR \
  --epochs $EPOCHS \
  --patience $PATIENCE \
  --lr $LEARNING_RATE \
  --batch_size $BATCH_SIZE \
  --beta $BETA \
  --save_dir $SAVE_DIR \
  --log_file $LOG_FILE > output.log 2>&1 &

```

## Training the diffusion model
Use the following to train the diffusion model
```
python train_ms.py --root_dir $DATA_PATH \
                    --csv_file $TRAINING_FILE \
                    --batch_size $BATCH_SIZE \
                    --lr 1e-4 \
                    --save_interval 500 \
                    --log_interval 10 \
                    --ema_rate 0.9999 \
                    --segmentation_model $UNET_PATH
```

## Generating synthetic data
Use the following to generate the synthetic data from artificial masks
```
python create_samples.py 
            --root_dir $root_dir
            --csv_file $csv_file
            --batch_size $batch_size
            --checkpoint $checkpoint

```