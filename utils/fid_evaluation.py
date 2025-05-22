# coding:UTF-8
# RuHe  2025/5/22 11:26
import math
import os
import numpy as np
import torch
import yaml

from einops import rearrange, repeat
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.mnist_dataset import MnistDataset
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class FIDEvaluation:
    def __init__(self, batch_size, config_path, accelerator=None, stats_dir='./results',
                 num_fid_samples=50000, inception_block_idx=2048):
        self.batch_size = batch_size
        self.print_fn = print if accelerator is None else accelerator.print
        self.state_dir = stats_dir
        self.device = device
        self.n_samples = num_fid_samples

        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.config = config
        self.model_config = config['model_params']
        self.train_config = config['train_params']
        self.diffusion_config = config['diffusion_params']
        self.dataset_config = config['dataset_params']

        # Load model
        self.model = Unet(self.model_config).to(self.device)
        ckpt_path = os.path.join(self.train_config['task_name'], self.train_config['ckpt_name'])
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

        # Load scheduler
        self.scheduler = LinearNoiseScheduler(
            num_time_steps=self.diffusion_config['num_time_steps'],
            beta_start=self.diffusion_config['beta_start'],
            beta_end=self.diffusion_config['beta_end']
        )

        # Inception model
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx]).to(device)
        self.dataset_stats_loaded = False
        self.channels = self.model_config['im_channels']

    def calculate_inception_features(self, samples):
        if self.channels == 1:
            samples = repeat(samples, 'b 1 ... -> b c ...', c=3)

        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]

        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))

        features = rearrange(features, '... 1 1 -> ...')
        return features

    def load_or_precalculate_dataset_stats(self):
        path = os.path.join(self.state_dir, 'dataset_stats')
        os.makedirs(path, exist_ok=True)
        try:
            ckpt = np.load(path + '.npz')
            self.m2, self.s2 = ckpt['m2'], ckpt['s2']
            self.print_fn('Dataset stats loaded from disk.')
            ckpt.close()
        except OSError:
            num_batches = int(math.ceil(self.n_samples / self.batch_size))
            stacked_real_features = []
            self.print_fn(f"Stacking Inception features for {self.n_samples} samples from the real dataset.")
            mnist = MnistDataset('train', im_path=self.dataset_config['im_path'])
            mnist_loader = DataLoader(mnist, batch_size=self.train_config['batch_size'], shuffle=True, num_workers=4)
            mnist_iter = iter(mnist_loader)
            for _ in tqdm(range(num_batches)):
                try:
                    real_samples = next(mnist_iter)
                except StopIteration:
                    break
                real_samples = real_samples.to(self.device)
                real_features = self.calculate_inception_features(real_samples)
                stacked_real_features.append(real_features)
            stacked_real_features = torch.cat(stacked_real_features, dim=0).cpu().numpy()
            m2 = np.mean(stacked_real_features, axis=0)
            s2 = np.cov(stacked_real_features, rowvar=False)
            np.savez_compressed(path, m2=m2, s2=s2)
            self.print_fn(f"Dataset stats cached to {path}.npz for future use.")
            self.m2, self.s2 = m2, s2
        self.dataset_stats_loaded = True

    @torch.inference_mode()
    def fid_score(self):
        if not self.dataset_stats_loaded:
            self.load_or_precalculate_dataset_stats()
        batches = num_to_groups(self.n_samples, self.batch_size)
        stacked_fake_features = []
        self.print_fn(f'Stacking Inception features for {self.n_samples} generated samples.')
        for batch in tqdm(batches):
            fake_samples = self.sample(batch_size=batch)
            fake_features = self.calculate_inception_features(fake_samples)
            stacked_fake_features.append(fake_features)
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
        m1 = np.mean(stacked_fake_features, axis=0)
        s1 = np.cov(stacked_fake_features, rowvar=False)

        return calculate_frechet_distance(m1, s1, self.m2, self.s2)

    def sample(self, batch_size):
        xt = torch.randn((batch_size, self.model_config['im_channels'],
                          self.model_config['im_size'], self.model_config['im_size'])).to(device)
        for i in reversed(range(self.diffusion_config['num_time_steps'])):
            noise_pred = self.model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
            xt, x0_pred = self.scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # 最终采样结果 [N, C, H, W]
        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        return ims


if __name__ == '__main__':
    evaluator = FIDEvaluation(batch_size=64, config_path='config/default.yaml')

    fid_value = evaluator.fid_score()
    print(f'FID: {fid_value:.2f}')
