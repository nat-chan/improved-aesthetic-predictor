import webdataset as wds
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import json

from warnings import filterwarnings


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

from os.path import join
from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json

import clip


from PIL import Image, ImageFile

import argparse
import sys
from pathlib import Path


# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

s = torch.load("sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo

model.load_state_dict(s)

model.to("cuda")
model.eval()


device = "cuda" if torch.cuda.is_available() else "cpu"
model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   

#####  This script will predict the aesthetic score for this image file:
parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='')
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--both', type=bool, default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num', type=int, default=0)
args = parser.parse_args()
if args.num == 0:
    if args.img == "":
        img_paths = [line.strip() for line in sys.stdin]
    elif args.img == "dump":
        img_paths, dst_paths = zip(*[line.strip().split() for line in sys.stdin])
    else:
        img_paths = [args.img]
    assert len(img_paths)%args.batchsize == 0
    pbar = tqdm(total=len(img_paths), dynamic_ncols=True)
    for j in range(len(img_paths) // args.batchsize):
        image = torch.cat([preprocess(Image.open(img_path)).unsqueeze(0).to(device) for img_path in img_paths[j * args.batchsize:(j + 1) * args.batchsize]])
        with torch.no_grad():
            image_features = model2.encode_image(image)
        im_emb_arr = normalized(image_features.cpu().detach().numpy() )
        input_tensor = torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor)
        prediction = model(input_tensor)
        for i in range(args.batchsize):
            if args.img == "dump":
                Path(dst_paths[j*args.batchsize+i]).write_text(str(float(prediction[i])))
            else:
                if args.both:
                    print( float(prediction[i]), img_paths[j*args.batchsize+i])
                else:
                    print( float(prediction[i]) )
        pbar.update(n=args.batchsize)
else:
    from nokogiri.working_dir import working_dir
    with working_dir("/home/natsuki/stylegan2-ada-pytorch"):
        from script_util import wrap_G
        G = wrap_G("/data/natsuki/training116/00023-white_yc05_yw04-mirror-auto4-gamma10-noaug/network-snapshot-021800.pkl")
    assert args.num%args.batchsize == 0
    pbar = tqdm(total=args.num, dynamic_ncols=True)
    for j in range(args.num // args.batchsize):
        image = torch.cat([
            preprocess(pil_img).unsqueeze(0).to(device)
            for pil_img in G.synth(G.map(range(args.batchsize*j+args.seed, args.batchsize*(j+1)+args.seed)))
        ])
        with torch.no_grad():
            image_features = model2.encode_image(image)
        im_emb_arr = normalized(image_features.cpu().detach().numpy() )
        input_tensor = torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor)
        prediction = model(input_tensor)
        for i in range(args.batchsize):
            print( float(prediction[i]) , args.batchsize*j+args.seed+i)
        pbar.update(n=args.batchsize)