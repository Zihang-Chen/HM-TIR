import os
import time
import numpy as np
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils as vutils
from torch.utils.data import DataLoader

from models.restormer_arch import Restormer_PPFN

from dataset import TrainSetLoader, TestSetLoader
import argparse
import utils
from utils import network_parameters
from warmup_scheduler import GradualWarmupScheduler

parser = argparse.ArgumentParser(description='Hyper-parameters')
parser.add_argument("--dataset_name", default='HM-TIR', type=str, help="dataset_name")
parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")
parser.add_argument("--save_dir", default='./exps', type=str, help="Save path of checkpoints")
parser.add_argument("--batch_size", type=int, default=16, help="Training batch sizse")
parser.add_argument("--patch_size", type=int, default=16, help="Training patch size")
parser.add_argument("--acc_step", type=int, default=1, help="Training accelerate step")
parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
parser.add_argument("--val_epoch", type=int, default=5, help="Validation of epoch")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: Adam, Adagrad, SGD")
parser.add_argument("--optimizer_settings", default={'lr_initial': 8e-5, 'lr_min': 1e-6}, type=dict,
                    help="optimizer settings")
parser.add_argument("--resume", default=False, type=bool, help="use resumed model parameters")
args = parser.parse_args()

## Set Seeds
seed = args.seed
torch.backends.cudnn.benchmark = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


OPT = {'BATCH': args.batch_size, 'EPOCHS': args.epochs, 'LR_INITIAL': args.optimizer_settings['lr_initial'],
       'LR_MIN': args.optimizer_settings['lr_min'], 'PATCH': args.patch_size,}


## Build Model
print('==> Build the model')
model_restored = Restormer_PPFN(inp_channels=1, out_channels=1)
r_number = network_parameters(model_restored)
type_npl = np.load(os.path.join(os.getcwd(), 'data/type7.npy')).astype(np.float32)
typep = torch.from_numpy(type_npl[:3]).clone().cuda()
multi_npl = np.load(os.path.join(os.getcwd(), 'data/multi2.npy')).astype(np.float32)
multip = torch.from_numpy(multi_npl).clone().cuda()
model_restored.cuda()

## Training model path direction
mode = 'Restormer-PPFN_' + args.dataset_name +'_deg-simple_loss-l1'

model_dir = os.path.join(args.save_dir, mode, 'models')
utils.mkdir(model_dir)

## GPU
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model_restored = nn.DataParallel(model_restored, device_ids=device_ids)

## Optimizer
start_epoch = 1
acc_step = args.acc_step
lr = float(OPT['LR_INITIAL'])
r_optimizer = optim.Adam([
    {'params': model_restored.parameters()}
],
    lr=lr, betas=(0.9, 0.999), eps=1e-8)

## Scheduler (Strategy)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(r_optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(r_optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

## Resume (Continue training by a pretrained model)
if args.resume:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restored, path_chk_rest, 'state_dict')
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(r_optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')

## Loss
L1loss = nn.L1Loss()

## DataLoaders
print('==> Loading datasets')

train_dataset = TrainSetLoader(args.dataset_dir, args.dataset_name, (OPT['PATCH'], OPT['PATCH']))
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'], shuffle=True)
val_dataset = TestSetLoader(args.dataset_dir, args.dataset_name, "composited", "hard")
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Model parameters:   {r_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    GPU:                {'GPU' + str(device_ids)}''')
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_psnr = 0
best_ssim = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
total_start_time = time.time()

## Log
log_dir = os.path.join(args.save_dir, mode, 'log')
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')
for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_r_loss = []
    epoch_p_loss = []

    model_restored.train()
    r_optimizer.zero_grad()
    tbar = tqdm(train_loader)
    for i, data in enumerate(tbar):
        target = data[1].cuda()
        input_ = data[0].cuda()
        m = data[2].cuda()

        total_loss = 0.0
        input_s = input_[:, -1]
        for idx in range(3):
            restored = model_restored([input_s, typep[2 - idx].unsqueeze(0), multip[m]])
            tar = (input_[:, 1 - idx] if idx < 2 else target) * (m == 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + target * (m == 0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # Compute loss
            loss_each = 1.0 * L1loss(restored, tar)
            if idx < 2:
                input_s = restored.clamp(0, 1).detach() * (m == 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + input_[:, 1 - idx] * (m == 0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            total_loss += loss_each.detach().cpu().item()

            loss = loss_each / acc_step
            loss.backward()

        # Back propagation
        if (i + 1) % acc_step == 0 or i == len(train_loader) - 1:
            r_optimizer.step()
            r_optimizer.zero_grad()
        epoch_r_loss.append(total_loss)
        tbar.set_description("Epoch: %d Restoration: loss = %f" % (epoch, np.mean(epoch_r_loss)))
    r_optimizer.zero_grad()


    ## Evaluation (Validation)
    if epoch % args.val_epoch == 0:
        model_restored.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []
        err_pos_list = []
        tbar = tqdm(val_loader)
        for ii, data_val in enumerate(tbar):
            N, C, H, W = data_val[1].shape
            with torch.no_grad():
                target = data_val[1].cuda()
                input_ = data_val[0].cuda()

                restored = input_
                for idx in range(3):
                    restored = model_restored([restored, typep[2-idx].unsqueeze(0), multip[1]])
                    restored = torch.clamp(restored, 0, 1)

                for res, tar in zip(restored, target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))
                    ssim_val_rgb.append(utils.torchSSIM(restored, target))

                tbar.set_description("psnr = %.4f, ssim = %.4f" % (torch.stack(psnr_val_rgb).mean().item(), torch.stack(ssim_val_rgb).mean().item()))
        img_grid_i = vutils.make_grid(input_[0], normalize=True, scale_each=True, nrow=8)
        writer.add_image('input img', img_grid_i, global_step=epoch)  # j 表示feature map数
        img_grid_o = vutils.make_grid(restored[0], normalize=True, scale_each=True, nrow=8)
        writer.add_image('output img', img_grid_o, global_step=epoch)  # j 表示feature map数
        img_gt = vutils.make_grid(target[0], normalize=True, scale_each=True, nrow=8)
        writer.add_image('img gt', img_gt, global_step=epoch)  # j 表示feature map数

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()

        # Save the best PSNR model of validation
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch_psnr = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': r_optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestPSNR.pth"))
        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
            epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))

        # Save the best SSIM model of validation
        if ssim_val_rgb > best_ssim:
            best_ssim = ssim_val_rgb
            best_epoch_ssim = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': r_optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestSSIM.pth"))
        print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
            epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))

        writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)
        writer.add_scalar('val/SSIM', ssim_val_rgb, epoch)
    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              np.mean(epoch_r_loss), scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': r_optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    writer.add_scalar('train/loss', np.mean(epoch_r_loss), epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
writer.close()

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
