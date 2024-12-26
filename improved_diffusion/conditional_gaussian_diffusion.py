import torch as th
import torch.nn.functional as F
from monai.metrics import HausdorffDistanceMetric
from .respace import SpacedDiffusion
from .gaussian_diffusion import LossType, ModelMeanType



class ConditionalGaussianDiffusion(SpacedDiffusion):
    def __init__(self, betas, model_mean_type,
                 model_var_type, loss_type=LossType.MSE, rescale_timesteps=False,
                 use_timesteps=None,
                 segmentation_model=None):
            super().__init__(
                  betas=betas,
                  model_mean_type=model_mean_type,
                  model_var_type=model_var_type,
                  loss_type=loss_type,
                  rescale_timesteps=rescale_timesteps,
                  use_timesteps=use_timesteps,
            )
            self.segmentation_model = segmentation_model
            self.hd = HausdorffDistanceMetric(include_background=False,  # Exclude background from the computation
            percentile=95,  # 95th percentile of distances
            reduction="mean",  # Reduce the metric by computing the mean
            get_not_nans=False  # Return mean over all values
        )
            
    
    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        abduct = False,
    ):
        """
        Generate samples from the model using DDIM, modified for counterfactual estimation
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            abduct=abduct
        ):
            final = sample
        return final["sample"]
    
    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        abduct=False,
    ):
        """
        Yield DDIM samples from each step, modified for counterfactual estimation
        """
        if device is None:
            device = next(model.parameters()).device

        assert isinstance(shape, (tuple, list))

        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        indices = list(range(self.num_timesteps)) # 0 -> T

        if not abduct:
            indices = indices[::-1] # T->0
            assert noise is not None, "Abduction Requires a input image as noise"

        if progress :
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i]*shape[0], device=device)
            with th.no_grad():
                if abduct:
                    out = self.ddim_reverse_sample(
                        model,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        eta=eta
                    )
                else:
                    out = self.ddim_sample(
                        model,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        eta=eta
                    )
                yield out
                img = out["sample"]


    def training_losses(self, model, x_start, t, noise=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        
        noise = noise if noise is not None else th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        drop_condition = model_kwargs.pop('drop_condition', False)
        model_kwargs = self._apply_drop_condition(model_kwargs, drop_condition)
        ground_truth = model_kwargs.pop('ground_truth')    
        
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        if self.loss_type == LossType.MSE:
            target = self._get_target(model_output, x_start, x_t, t, noise)
            diffusion_loss = th.mean((model_output - target) ** 2)
        else:
             raise NotImplementedError(f"Unsupported Loss Type : {self.loss_type}")
        
        total_loss = diffusion_loss

        if self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x_t, t, model_output)
        elif self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        else:
            raise NotImplementedError()

        if self.segmentation_model is not None:
            with th.no_grad():
                predicted_mask = self.segmentation_model(pred_xstart)
            segmentation_loss =  th.nn.functional.mse_loss(predicted_mask, ground_truth)
            contour_loss = self._contour_loss(predicted_mask, ground_truth)
            # hausdorff_loss = self._hausdorff_loss(predicted_mask, ground_truth)
            # total_loss += segmentation_loss + contour_loss + hausdorff_loss
            total_loss += segmentation_loss + contour_loss

        return {"loss": total_loss}


    def _apply_drop_condition(self, model_kwargs, drop_condition):
        if drop_condition:
            # set the mask to random but keep the other variables as is for loss computation
            model_kwargs['mask'] = th.randn_like(model_kwargs.get('ground_truth'))
        else:
            model_kwargs['mask'] = model_kwargs.get('ground_truth')
        return model_kwargs
    
    def _get_target(self, model_output, x_start, x_t, t, noise):
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            return self.q_posterior_mean_variance(x_start, x_t, t)[0]
        elif self.model_mean_type == ModelMeanType.START_X:
            return x_start
        elif self.model_mean_type == ModelMeanType.EPSILON:
            return noise
        else:
            raise NotImplementedError(self.model_mean_type)
    
    def _contour_loss(self, predicted, ground_truth):
        # Sobel filter for detecting edges (contours)
        sobel_x = th.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0)
        sobel_y = th.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0)

        sobel_x = sobel_x.to(predicted.device)
        sobel_y = sobel_y.to(predicted.device)

        # Check if both masks are empty (all zeros)
        if th.sum(predicted) == 0 and th.sum(ground_truth) == 0:
            return th.tensor(0.0, device=predicted.device)  # Return 0 for empty masks
    
        # Apply the Sobel filter to find edges
        predicted_edges_x = F.conv2d(predicted, sobel_x, padding=1)
        predicted_edges_y = F.conv2d(predicted, sobel_y, padding=1)
        predicted_edges = th.sqrt(predicted_edges_x**2 + predicted_edges_y**2)

        ground_truth_edges_x = F.conv2d(ground_truth, sobel_x, padding=1)
        ground_truth_edges_y = F.conv2d(ground_truth, sobel_y, padding=1)
        ground_truth_edges = th.sqrt(ground_truth_edges_x**2 + ground_truth_edges_y**2)

        # Calculate the MSE between the edges
        loss = F.mse_loss(predicted_edges, ground_truth_edges)
        return loss
    
    
    def _hausdorff_loss(self, predicted, ground_truth):
        # Ensure binary masks
        predicted_bin = (predicted > 0.5).float()
        ground_truth_bin = (ground_truth > 0.5).float()
        if th.sum(predicted_bin) == 0 and th.sum(ground_truth_bin) == 0:
            return th.tensor(0.0, device=predicted.device)  # Return 0 for empty masks
        # Reset the metric state to ensure clean calculation
        self.hd.reset()

        # Add the predictions and ground truth to the metric
        self.hd(y_pred=predicted_bin, y=ground_truth_bin)

        # Compute and return the Hausdorff distance
        hausdorff_dist = self.hd.aggregate().mean()

        return hausdorff_dist
