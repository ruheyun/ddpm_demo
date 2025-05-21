# coding:UTF-8
# RuHe  2025/5/20 20:14
import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, model_config, diffusion_config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions.
    :param model:
    :param scheduler:
    :param train_config:
    :param model_config:
    :param diffusion_config:
    :return:
    """
    xt = torch.randn((train_config['num_samples'], model_config['im_channels'],
                      model_config['im_size'], model_config['im_size'])).to(device)
    for i in tqdm(reversed(range(diffusion_config['num_time_steps']))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # Save x0
        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)
        fill_path = os.path.join(train_config['task_name'], 'samples')
        if not os.path.exists(fill_path):
            os.mkdir(fill_path)
        img.save(os.path.join(fill_path, f'x0_{i}.png'))
        img.close()


def infer(args):
    # Read the config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # Load model with checkpoint
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['ckpt_name']), map_location=device))
    model.eval()

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_time_steps=diffusion_config['num_time_steps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path', default='config/default.yaml', type=str)
    args_p = parser.parse_args()
    infer(args_p)
