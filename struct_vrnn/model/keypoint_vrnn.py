import torch
import torch.nn as nn

from hydra.utils import instantiate


def compute_temporal_separation_loss(keypoints, sigma):
    batch_size, time, num_keypoints, _ = keypoints.shape
    x_coords = keypoints[..., 0]
    y_coords = keypoints[..., 1]
    x_coords_centered = x_coords - x_coords.mean(dim=1, keepdim=True)
    y_coords_centered = y_coords - y_coords.mean(dim=1, keepdim=True)
    distances = (x_coords_centered[..., None] - x_coords_centered[..., None, :]) ** 2 + \
                (y_coords_centered[..., None] - y_coords_centered[..., None, :]) ** 2
    distances = distances.mean(dim=1)
    loss_matrix = torch.exp(-distances / (2 * sigma ** 2))
    loss_matrix = loss_matrix.mean(dim=0)
    loss = loss_matrix.sum()
    loss -= num_keypoints
    loss /= num_keypoints * (num_keypoints - 1)
    return loss


def compute_sparsity_loss(keypoints):
    # Input: keypoints (batch_size, time, num_keypoints, 3)
    # Return: sum of l1 norm of keypoint intensities
    intensities = keypoints[..., 2]
    loss = intensities.abs().sum(dim=-1).mean()
    return loss


class StructVRNN(nn.Module):

    def __init__(self, keypoint_detector, dynamics, temporal_separation_loss_sigma):
        super().__init__()
        self.keypoint_detector = keypoint_detector
        self.dynamics = dynamics
        self.temporal_separation_loss_sigma = temporal_separation_loss_sigma

    def forward(self, videos, actions):
        # Perform forward prediction.
        sequence_length = actions.shape[1]
        # Flatten video sequence into batch of images.
        images = videos.reshape((-1,) + videos.shape[2:])
        # Keypoint detection/reconstruction training.
        observed_keypoints = self.keypoint_detector.image_to_keypoints(images)
        # Add time dimension back to observed_keypoints.
        observed_keypoints_time = observed_keypoints.view(videos.shape[0], videos.shape[1], -1, 3)

        first_frame_keypoints = observed_keypoints_time[:, 0]
        first_frame_keypoints_rep = torch.repeat_interleave(first_frame_keypoints, sequence_length, dim=0)
        first_frame = torch.repeat_interleave(videos[:, 0], sequence_length, dim=0)
        # Dynamics training.
        # Reshape keypoints into temporal sequence.
        observed_keypoints = observed_keypoints.view(videos.shape[:2] + observed_keypoints.shape[1:])
        # Compute sparsity loss
        keypoint_preds_stacked = self.dynamics.predict(observed_keypoints, actions)
        # Add num_keypoints dimension back to keypoint_preds.
        keypoint_preds = keypoint_preds_stacked.view(videos.shape[0], -1, self.num_keypoints, 3)
        # reconstructed
        # flatten batch*time dimension of keypoint_preds
        reconstructed_predictions = self.keypoint_detector.keypoints_to_image(
            keypoint_preds.view(-1, self.num_keypoints, 3), first_frame, first_frame_keypoints_rep)
        reconstructed_predictions = reconstructed_predictions.view(videos.shape[0], sequence_length, *videos.shape[2:])
        return reconstructed_predictions

    @property
    def num_keypoints(self):
        return self.keypoint_detector.num_keypoints

    def train_step(self, videos, actions, reconstruct_forward_predictions=False):
        sequence_length = videos.shape[1]
        # Flatten video sequence into batch of images.
        images = videos.view((-1,) + videos.shape[2:])
        # Keypoint detection/reconstruction training.
        observed_keypoints = self.keypoint_detector.image_to_keypoints(images)
        # Add time dimension back to observed_keypoints.
        observed_keypoints_time = observed_keypoints.view(videos.shape[0], sequence_length, -1, 3)

        first_frame_keypoints = observed_keypoints_time[:, 0]

        # Visualize keypoints
        # from struct_vrnn.viz_utils import visualize_keypoints
        # import cv2
        # import numpy as np
        # # visualize keypoints and save to file
        # keypoints_vis = np.concatenate([visualize_keypoints(observed_keypoints_time[0, i].detach().cpu().numpy(), 64) for i in range(10)], axis=1)
        # cv2.imwrite('keypoints.png', keypoints_vis)

        first_frame_keypoints_rep = torch.repeat_interleave(first_frame_keypoints, sequence_length, dim=0)

        first_frame = torch.repeat_interleave(videos[:, 0], sequence_length, dim=0)
        reconstructed_images = self.keypoint_detector.keypoints_to_image(observed_keypoints, first_frame,
                                                                         first_frame_keypoints_rep)
        # Dynamics training.
        # Reshape keypoints into temporal sequence.
        observed_keypoints = observed_keypoints.view(videos.shape[:2] + observed_keypoints.shape[1:])
        # Compute sparsity loss
        sparsity_loss = compute_sparsity_loss(observed_keypoints)
        # Detach keypoints for dynamics computation.
        observed_keypoints_detached = observed_keypoints.detach()
        keypoint_preds_stacked, klds = self.dynamics(observed_keypoints_detached, actions)
        # Add num_keypoints dimension back to keypoint_preds.
        keypoint_preds = keypoint_preds_stacked.view(videos.shape[0], -1, self.num_keypoints, 3)
        # Compute loss.
        reconstruction_loss = nn.functional.mse_loss(reconstructed_images, images)
        temporal_separation_loss = compute_temporal_separation_loss(keypoint_preds, self.temporal_separation_loss_sigma)
        coord_pred_loss = ((keypoint_preds - observed_keypoints_detached[:, 1:]) ** 2).sum(dim=(-1, -2)).mean(
            dim=(0, 1))

        kld_loss = klds.mean()

        if reconstruct_forward_predictions:
            first_frame_keypoints_rep_p = torch.repeat_interleave(first_frame_keypoints, sequence_length - 1, dim=0)
            first_frame_p = torch.repeat_interleave(videos[:, 0], sequence_length - 1, dim=0)
            reconstructed_predictions = self.keypoint_detector.keypoints_to_image(
                keypoint_preds.view(-1, self.num_keypoints, 3), first_frame_p, first_frame_keypoints_rep_p)
            reconstructed_predictions = reconstructed_predictions.view(videos.shape[0], sequence_length - 1,
                                                                       *videos.shape[2:])
            reconstructed_images = reconstructed_images.view(videos.shape)
        else:
            reconstructed_predictions = None

        return reconstructed_images, \
               observed_keypoints_detached, \
               reconstructed_predictions, \
               dict(
                   reconstruction_loss=reconstruction_loss,
                   temporal_separation_loss=temporal_separation_loss,
                   coord_pred_loss=coord_pred_loss,
                   sparsity_loss=sparsity_loss,
                   kld_loss=kld_loss,
               )
