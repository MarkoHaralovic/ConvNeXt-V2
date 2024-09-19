import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import argparse
import numpy as np
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm.optim.optim_factory as optim_factory

from datasets import build_dataset
import models.fcmae as fcmae

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import str2bool

def get_args_parser():
    parser = argparse.ArgumentParser('FCMAE fine-tuning', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')
    
    parser.add_argument('--mask_ratio', default=0.6, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--decoder_depth', type=int, default=1)
    parser.add_argument('--decoder_embed_dim', type=int, default=512)

    # Model parameters
    parser.add_argument('--model', default='convnextv2_atto', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--drop_path', type=float, default=0., metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--layer_decay_type', type=str, choices=['single', 'group'], default='single',
                        help="""Layer decay strategies. The single strategy assigns a distinct decaying value for each layer,
                        whereas the group strategy assigns the same decaying value for three consecutive layers""")
    
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=0.0004, metavar='LR',
                        help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--data_path', default='/imagenet/train', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='visualization',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='pretrained_weights/checkpoint-799.pth',
                        help='resume from checkpoint')
    
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--auto_resume', type=str2bool, default=True)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--patch_size', type=int, default=32)

    return parser

def denormalize(img_bchw,image_mean,image_std):
    """
    Denormalize the image for correct visualization.
    """
    return img_bchw.mul(image_std).add_(image_mean).clamp_(0., 1.)

def main(args):
    print(args)
    device = torch.device(args.device)
    IMAGENET_RGB_MEAN = torch.tensor((0.485, 0.456, 0.406), device=device).reshape(1, 3, 1, 1)
    IMAGENET_RGB_STD = torch.tensor((0.229, 0.224, 0.225), device=device).reshape(1, 3, 1, 1)
    cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.Resize((args.input_size,args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path), transform=transform)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem
    )
    dataiter = iter(data_loader_train)

 
    # Create the FCMAE model
    fcmae_model = fcmae.__dict__[args.model](
        img_size=args.input_size, 
        patch_size=args.patch_size, 
        mask_ratio=args.mask_ratio,
        decoder_depth=args.decoder_depth, 
        decoder_embed_dim=args.decoder_embed_dim, 
        norm_pix_loss=args.norm_pix_loss
    )

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    fcmae_model = fcmae_model.to(device)

    model_without_ddp = fcmae_model

    distributed = False
    if distributed:
        fcmae_model = torch.nn.parallel.DistributedDataParallel(fcmae_model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = fcmae_model.module

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    utils.auto_load_model(
        args=args, 
        model=fcmae_model, 
        model_without_ddp=model_without_ddp,
        optimizer=optimizer, 
        loss_scaler=loss_scaler
    )   
    fcmae_model = fcmae_model.to(device)
    fcmae_model.eval()

    i = 0 
    try :
        while True:
            img, _ = next(dataiter)
            img = img.to(device)

            loss, predictions, mask = fcmae_model(img)
            mask = mask.unsqueeze(-1).repeat(1, 1, (args.patch_size**2 * 3))

            predictions = predictions.view(1, args.patch_size**2 * 3, -1).permute(0, 2, 1)
            predictions = fcmae_model.unpatchify(predictions )
            
            mask = fcmae_model.unpatchify(mask).to(device)
            
            im_masked = img * (1 - mask)
            rec_and_img = predictions * mask + im_masked
            channel_means = img.mean(dim=[2, 3], keepdim=True)
            im_mean_in = img * (1 - mask) + mask * channel_means
            
            # Denormalizing the images before visualization
            pred_img = denormalize(predictions * mask,image_mean=IMAGENET_RGB_MEAN,image_std=IMAGENET_RGB_STD)[0].permute(1, 2, 0).cpu().detach().numpy()
            masked_img = denormalize(im_masked,image_mean=IMAGENET_RGB_MEAN,image_std=IMAGENET_RGB_STD)[0].permute(1, 2, 0).cpu().detach().numpy()
            rec_img = denormalize(rec_and_img,image_mean=IMAGENET_RGB_MEAN,image_std=IMAGENET_RGB_STD)[0].permute(1, 2, 0).cpu().detach().numpy()
            im_mean_in = denormalize(im_mean_in,image_mean=IMAGENET_RGB_MEAN,image_std=IMAGENET_RGB_STD)[0].permute(1, 2, 0).cpu().detach().numpy()
            orig_img = denormalize(img,image_mean=IMAGENET_RGB_MEAN,image_std=IMAGENET_RGB_STD)[0].permute(1, 2, 0).cpu().numpy()

            # Plot the original, masked, and reconstructed images
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 5, 1)
            plt.title("Original Image",fontsize=8)
            plt.imshow(orig_img)
            plt.axis('off')

            plt.subplot(1, 5, 2)
            plt.title("Masked Image",fontsize=8)
            plt.imshow(masked_img)
            plt.axis('off')

            plt.subplot(1, 5, 3)
            plt.title("Reconstructed Image",fontsize=8)
            plt.imshow(pred_img)
            plt.axis('off')

            plt.subplot(1, 5, 4)
            plt.title("Reconstructed + Masked Image",fontsize=8)
            plt.imshow(rec_img)
            plt.axis('off')

            plt.subplot(1, 5, 5)
            plt.title("Mean + Masked Image",fontsize=8)
            plt.imshow(im_mean_in)
            plt.axis('off')

            plt.savefig(f'{args.output_dir}/visual_reconstruction{i}.png')
            plt.show()

            i+=1
    except StopIteration:
        return

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)