from tqdm.auto import tqdm
import torch
import numpy as np
import torchvision


class DDPM:

    def __init__(self, config):
        self.config = config
        self.steps = config['diffusion']['steps']
        self.beta_schedule = config['diffusion']['beta_schedule']
        self.betas = self.beta_scheduler(self.beta_schedule, self.steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)

    def beta_scheduler(self, name, steps):
        if name == 'linear':
            return np.linspace(0.1 / steps, 20 / steps, steps)
        elif name == "cosine":
            f = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
            t = np.arange(0, 1, 1 / steps)
            return np.clip(1 - f(t + 1 / steps) / f(t), 0, 0.999)
    
    def q_distribution(self, x_0, t):
        """
        Get the distribution q(x_t | x_0).
        """
        mean = self.alphas_cumprod[t] ** 0.5 * x_0
        var = 1 - self.alphas_cumprod[t]
        return mean, var

    def q_sample(self, x_0, t):
        """
        Sample from distribution q(x_t | x_0).
        """
        if t: noise = torch.randn_like(x_0)
        else: noise = torch.zeros_like(x_0)
        mean, var = self.q_distribution(x_0, t)
        return mean + var ** 0.5 * noise

    def q_posterior_distribution(self, x_t, x_0, t):
        """
        Get the distribution q(x_{t-1} | x_t, x_0).
        """
        alpha_cumprod_prev = self.alphas_cumprod[t - 1] if t - 1 else 1
        w_t = self.alphas[t] ** 0.5 * (1 - alpha_cumprod_prev)
        w_0 = alpha_cumprod_prev ** 0.5 * self.betas[t]
        mean = (w_t * x_t + w_0 * x_0) / (1 - self.alphas_cumprod[t])
        var = (1 - alpha_cumprod_prev) * self.betas[t] / (1 - self.alphas_cumprod[t])
        return mean, var

    def predict_x_0(self, x_t, eps, t):
        x_0_hat = (x_t - (1 - self.alphas_cumprod[t]) ** 0.5 * eps)
        x_0_hat = (x_0_hat / self.alphas_cumprod[t] ** 0.5).clamp(-1, 1)
        return x_0_hat

    def sample(self, x_T, y_n, model, conditioner):
        """
        The function used for sampling from noise.
        """ 
        x_t = x_T
        x_t.requires_grad = True
        steps_pbar = tqdm(range(self.steps-1, -1, -1))
        for t in steps_pbar:
            eps = model(x_t, torch.tensor([t]*x_t.shape[0], device=x_t.device))[:,:3,:,:]
            x_0_hat = self.predict_x_0(x_t, eps, t)
            x_tp_mean, x_tp_var = self.q_posterior_distribution(x_t, x_0_hat, t)
            noise = torch.randn_like(x_t)
            if t: x_tp_mean += x_tp_var ** 0.5 * noise
            y_tn = self.q_sample(y_n, t)
            x_t, distance = conditioner(x_0_hat, x_tp_mean, x_t, y_n, y_tn)
            steps_pbar.set_postfix({'distance': distance.item()}, refresh=False)
        return x_t
    




