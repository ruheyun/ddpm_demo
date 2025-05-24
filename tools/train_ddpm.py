# coding:UTF-8
# RuHe  2025/5/20 18:10
import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler


# import torch_xla.core.xla_model as xm
# device = xm.xla_device()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    # read the config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_time_steps=diffusion_config['num_time_steps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    # Create the dataset
    mnist = MnistDataset('train', im_path=dataset_config['im_path'])
    mnist_loader = DataLoader(mnist, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)

    # Instantiate the model
    model = Unet(model_config).to(device)
    model.train()

    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    # Load checkpoint if found
    fill_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    if os.path.exists(fill_path):
        print('Loading checkpoint as found one.')
        model.load_state_dict(torch.load(fill_path, map_location=device))

    # Specify training parameters
    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()

    # Run training
    for epoch_idx in range(num_epochs):
        losses = []
        for im in tqdm(mnist_loader):
            optimizer.zero_grad()
            im = im.float().to(device)

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample time steps
            t = torch.randint(0, diffusion_config['num_time_steps'], (im.shape[0],)).to(device)

            # Add noise to images according to time steps
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.detach().cpu().item())
            loss.backward()
            optimizer.step()
            # xm.optimizer_step(optimizer)

        print(f'Finished epoch: {epoch_idx + 1} | Loss: {np.mean(losses): .4f}')
        torch.save(model.cpu().state_dict(), fill_path)

    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path', default='config/default.yaml', type=str)
    args_p = parser.parse_args()
    train(args_p)
