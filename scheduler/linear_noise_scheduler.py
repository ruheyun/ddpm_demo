# coding:UTF-8
# RuHe  2025/5/19 17:56
import torch


class LinearNoiseScheduler:
    r"""
    Class for the linear noise scheduler that is used DDPM
    """
    def __init__(self, num_time_steps, beta_start, beta_end):
        self.num_time_steps = num_time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_time_steps)
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1. - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor from normal dist
        :param t: timestep of the forward process of shape -> B
        :return:
        """
        original_shape = original.shape
        batch_size = original_shape[0]
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)

        # reshape till (B) becomes (B, 1, 1, 1) if image is (B, C, H, W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # apply and return forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)

    def sample_prev_timestep(self, x_t, noise_pred, t):
        r"""
        Use the noise prediction by model to get x_t-1 using x_t and the noise predicted
        :param x_t: current timestep sample
        :param noise_pred: model noise predicted
        :param t: current timestep we are at
        :return:
        """
        x_0 = ((x_t - self.sqrt_one_minus_alpha_cum_prod.to(x_t.device)[t] * noise_pred)
               / self.sqrt_alpha_cum_prod.to(x_t.device)[t])
        x_0 = torch.clamp(x_0, -1., 1.)

        mean = x_t - self.betas.to(x_t.device)[t] * noise_pred / self.sqrt_one_minus_alpha_cum_prod.to(x_t.device)[t]
        mean /= torch.sqrt(self.alphas.to(x_t.device)[t])

        if t == 0:
            return mean, x_0
        else:
            variance = (1. - self.alpha_cum_prod.to(x_t.device)[t - 1]) / (1. - self.alpha_cum_prod.to(x_t.device)[t])
            variance *= self.betas.to(x_t.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(x_t.shape).to(x_t.device)

            return mean + sigma * z, x_0
