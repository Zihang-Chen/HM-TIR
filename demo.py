import os
from collections import OrderedDict
import numpy as np
import pyiqa
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset import VisSetLoader
from models.restormer_arch import Restormer_PPFN

parser = argparse.ArgumentParser(description='Hyper-parameters')
parser.add_argument("--dataset_dir", default='./example/hard', type=str, help="train_dataset_dir")
parser.add_argument('--output', default='./example/enh_hard', type=str, help='Directory for results')
parser.add_argument('--weights', type=str, default='./checkpoint/best.pth', help='Path to weights')
args = parser.parse_args()


def save_img(filepath, img):
    img.save(filepath)


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        # print(state_dict)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


model = Restormer_PPFN(inp_channels=1, out_channels=1)

model.cuda()
load_checkpoint(model, args.weights)
model.eval()

out_dir = args.output

dataset = VisSetLoader(args.dataset_dir)
loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
type_npl = np.load(os.path.join(os.getcwd(), 'data/type7.npy')).astype(np.float32)
typep = torch.from_numpy(type_npl[:3]).clone().cuda()
multi_npl = np.load(os.path.join(os.getcwd(), 'data/multi2.npy')).astype(np.float32)
multip = torch.from_numpy(multi_npl).clone().cuda()
index = 0

tbar = tqdm(loader)
type_dic = {"contrast": 0, "blur": 1, "noise": 2}
sce_dic = {"single": 0, "composited": 1}
scenario_prompt = "composited"
type_prompt = ['noise', 'blur', 'contrast']
for ii, data_val in enumerate(tbar):
    input_ = data_val[0].cuda()
    file_name = data_val[2][0]
    h, w = data_val[1]

    with torch.no_grad():

        restored = input_
        for idx in type_prompt:
            restored = model([restored, typep[type_dic[idx]].unsqueeze(0), multip[sce_dic[scenario_prompt]]])
            restored = torch.clamp(restored, 0, 1)

        restored = transforms.ToPILImage()(restored[0, 0, :h, :w].detach().cpu())

    f = file_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_img((os.path.join(out_dir, f)), restored)
    index += 1

print(f"Files saved at {out_dir}")
print('finish !')
