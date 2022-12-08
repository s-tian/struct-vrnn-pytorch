import os
import numpy as np
import torch
from tqdm import tqdm

import omegaconf
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

import wandb
from struct_vrnn.viz_utils import visualize_keypoints


def batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {key: batch_to_device(value, device) for key, value in batch.items()}
    elif isinstance(batch, list) or isinstance(batch, tuple):
        return [batch_to_device(value, device) for value in batch]
    else:
        return batch.to(device)


def train_epoch(cfg, train_steps, model, optimizer, dataloader):
    model.train()
    pbar = tqdm(dataloader)
    step = train_steps
    for train_batch in pbar:
        if isinstance(train_batch, dict):
            train_batch = train_batch["video"].float(), train_batch["actions"].float()
        train_batch = batch_to_device(train_batch, cfg.device)
        optimizer.zero_grad()
        reconstructed_images, keypoints, reconstructed_preds, losses = model.train_step(*train_batch,
                                                                                        reconstruct_forward_predictions=step%cfg.log_images_every==0)
        loss = 0.0
        for loss_name, loss_item in losses.items():
            if cfg.training_weights.get(loss_name) is not None:
                loss += cfg.training_weights.get(loss_name) * loss_item
        losses["total_loss"] = loss
        wandb_log = losses
        if step % cfg.log_images_every == 0:
            # Log reconstructed images.
            gt_video = train_batch[0]
            reconstructed_images = torch.cat(list(reconstructed_images[:, -1]), dim=2)
            gt_video_horiz = torch.cat(list(gt_video[:, -1]), dim=2)
            gt_video_firstframe_horiz = torch.cat(list(gt_video[:, 0]), dim=2)
            keypoints = keypoints.cpu().detach().numpy()
            print("Keypoints are ", keypoints)
            keypoint_vis = [visualize_keypoints(keypoints[i, -1], gt_video.shape[-2]) for i in range(keypoints.shape[0])]
            keypoint_vis = torch.cat(list(torch.from_numpy(np.stack(keypoint_vis, axis=0)).permute(0, 3, 1, 2).float().cuda()), dim=2) / 255.
            reconstructed_images = torch.cat((gt_video_horiz, reconstructed_images, gt_video_firstframe_horiz, keypoint_vis), dim=-2)
            reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
            wandb_log["reconstructed_images"] = wandb.Image(reconstructed_images.detach().cpu() * 255)
            gt_and_pred_video = torch.cat([gt_video[:, 1:], reconstructed_preds], dim=-2)
            gt_and_pred_video = torch.cat(list(gt_and_pred_video[:cfg.num_videos_to_log]), dim=-1)
            # Clip to [0, 1] range.
            gt_and_pred_video = torch.clamp(gt_and_pred_video, 0.0, 1.0)
            wandb_log["gt_and_pred_video"] = wandb.Video(gt_and_pred_video.detach().cpu() * 255, fps=5)

        wandb.log(wandb_log)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {loss.item():.4f}")

        if step > 0 and step % cfg.log_every == 0:
            # Save torch checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
            }, f"checkpoint-{step}.pt")
        step += 1
        if step > train_steps + cfg.train_epoch_length:
            return


def val_epoch(cfg, model, dataloader):
    model.eval()
    pbar = tqdm(dataloader)
    step = 0
    wandb_log = dict()
    for train_batch in pbar:
        if isinstance(train_batch, dict):
            train_batch = train_batch["video"].float(), train_batch["actions"].float()
        train_batch = batch_to_device(train_batch, cfg.device)
        gt_video = train_batch[0]
        train_batch = train_batch[0][:, :2], train_batch[1]
        reconstructed_preds = model(*train_batch)
        if step % cfg.log_images_every == 0:
            # Log reconstructed images.
            gt_and_pred_video = torch.cat([gt_video[:, 1:], reconstructed_preds[:, :gt_video.shape[1]-1]], dim=-2)
            gt_and_pred_video = torch.cat(list(gt_and_pred_video[:cfg.num_videos_to_log]), dim=-1)
            # Clip to [0, 1] range.
            gt_and_pred_video = torch.clamp(gt_and_pred_video, 0.0, 1.0)
            wandb_log["val_gt_and_pred_video"] = wandb.Video(gt_and_pred_video.detach().cpu() * 255, fps=5)
        step += 1
        if step > cfg.val_steps:
            break
    wandb.log(wandb_log)


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg):
    print("Logging to ", os.getcwd())
    model = instantiate(cfg.model)
    model = model.to(cfg.device)

    if cfg.data_type == "dataset":
        train_data, val_data = instantiate(cfg.dataset, split="train"), instantiate(cfg.dataset, split="val")
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=cfg.num_workers)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, pin_memory=True, num_workers=cfg.num_workers)
    elif cfg.data_type == "robomimic":
        from fitvid.data.robomimic_data import load_dataset_robomimic_torch
        train_loader = load_dataset_robomimic_torch(
            dataset_path=cfg.dataset.dataset_files,
            batch_size=cfg.batch_size,
            video_len=cfg.dataset.num_steps,
            video_dims=(64, 64),
            phase="train",
            depth=False,
            normal=False,
            view=cfg.dataset.camera_name,
            cache_mode=cfg.dataset.cache_mode,
            seg=False,
        )
        val_loader = load_dataset_robomimic_torch(
            dataset_path=cfg.dataset.dataset_files,
            batch_size=cfg.batch_size,
            video_len=cfg.dataset.num_steps,
            video_dims=(64, 64),
            phase="valid",
            depth=False,
            normal=False,
            view=cfg.dataset.camera_name,
            cache_mode=cfg.dataset.cache_mode,
            seg=False,
        )
    elif cfg.data_type == "hdf5":
        train_data, val_data = instantiate(cfg.dataset, phase="train"), instantiate(cfg.dataset, phase="val")
        train_loader = train_data.get_data_loader(cfg.batch_size)
        val_loader = val_data.get_data_loader(cfg.batch_size)
    else:
        raise NotImplementedError(f"Unknown data type {cfg.data_type}")

    # Save config as yaml file to output directory
    omegaconf.OmegaConf.save(cfg, "config.yaml")

    # set run name to be the last two folders in the output directory
    run_name = "_".join(os.getcwd().split("/")[-2:])

    wandb.init(project="struct_vrnn", config=cfg, name=run_name)
    wandb.config.update(omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    wandb.config.slurm_job_id = os.getenv("SLURM_JOB_ID", 0)
    wandb.config.output_dir = os.getcwd()

    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    train_steps = 0
    while train_steps < cfg.train_steps:
        train_epoch(cfg, train_steps, model=model, dataloader=train_loader, optimizer=optimizer)
        train_steps += min(len(train_loader), cfg.train_epoch_length)
        with torch.no_grad():
            val_epoch(cfg, model=model, dataloader=val_loader)


if __name__ == "__main__":
    main()