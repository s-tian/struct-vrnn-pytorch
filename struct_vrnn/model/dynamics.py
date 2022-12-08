import torch
import torch.nn as nn

from hydra.utils import instantiate


def kl_divergence(mean1, logvar1, mean2, logvar2):
    kld = 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + torch.square(mean1 - mean2) * torch.exp(-logvar2)
    )
    return torch.mean(kld)


class Dynamics(nn.Module):

    def __init__(self, prior, posterior, dynamics):
        super().__init__()
        self.prior = prior
        self.posterior = posterior
        # The dynamics model is deterministic, but we use the Gaussian implementation and always take the mean.
        self.dynamics = dynamics

    def forward(self, keypoints, actions):
        dynamics_state, prior_state, posterior_state = None, None, None
        video_len = keypoints.shape[1]
        # Flatten keypoints
        keypoints = keypoints.view(keypoints.shape[:2] + (-1,))
        keypoint_preds, klds = list(), list()
        for t in range(1, video_len):
            k_t, k_t_1 = keypoints[:, t - 1], keypoints[:, t]
            (z_t, mu, logvar), posterior_state = self.posterior(k_t_1, posterior_state)
            (_, prior_mu, prior_logvar), prior_state = self.prior(k_t, prior_state)
            inp = torch.cat((k_t, actions[:, t - 1], z_t), dim=-1)
            (_, pred_mu, _), dynamics_state = self.dynamics(inp, dynamics_state)
            # Apply tanh to make sure that the predicted keypoints are in -1, 1
            pred_mu = torch.tanh(pred_mu)
            # compute KL(posterior || prior).
            klds.append(kl_divergence(mu, logvar, prior_mu, prior_logvar))
            keypoint_preds.append(pred_mu)
        keypoint_preds = torch.stack(keypoint_preds, dim=1)
        klds = torch.stack(klds)
        return keypoint_preds, klds

    def predict(self, keypoints, actions):
        dynamics_state, prior_state, posterior_state = None, None, None
        num_actions = actions.shape[1]
        num_context = keypoints.shape[1]
        # Flatten keypoints
        keypoints = keypoints.view(keypoints.shape[:2] + (-1,))
        keypoint_preds, klds = list(), list()
        for t in range(num_actions):
            if t < num_context:
                k_t = keypoints[:, t]
            else:
                k_t = keypoint_preds[-1]
            (z_t, prior_mu, prior_logvar), prior_state = self.prior(k_t, prior_state)
            inp = torch.cat((k_t, actions[:, t - 1], z_t), dim=-1)
            (_, pred_mu, _), dynamics_state = self.dynamics(inp, dynamics_state)
            # Apply tanh to make sure that the predicted keypoints are in -1, 1
            pred_mu = torch.tanh(pred_mu)
            # compute KL(posterior || prior).
            keypoint_preds.append(pred_mu)
        keypoint_preds = torch.stack(keypoint_preds, dim=1)
        return keypoint_preds