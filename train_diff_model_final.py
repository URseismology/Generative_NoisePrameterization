import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Subset
import pickle
import os
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage as EMA
from types import SimpleNamespace
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import sys
sys.path.insert(0, os.getcwd())
from PRJ_AUTO_NOISEPARAM.code.lib.diff_model_utils import *
from PRJ_AUTO_NOISEPARAM.code.lib.utils import *


def training_loop(loader  : DataLoader,
                  model   : nn.Module,
                  schedule: Schedule,
                  epochs  : int = 10000,
                  lr      : float = 1e-3):
    
    optimizer = optim.AdamW(model.parameters(),lr=lr)
    lrscheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=1300)

    loss_track = []
    model.train()

    for epoch in (pbar := tqdm(range(epochs))):
        for x0 in loader:
            x0 = x0[:, :1000].float().to("cuda")  # Truncate and ensure tensor format
            x0 = x0.unsqueeze(0).unsqueeze(0)
            data_resized = F.interpolate(x0, size=(128, 128), mode='area') 
            data_resized = data_resized.squeeze(0).squeeze(0)
            x0_covar = data_resized.T@data_resized   
            for i in range(0,len(x0_covar)):
                x0_covar[i] = x0_covar[i]/x0_covar[i,i]
            x0 = x0_covar
            
            x0 = x0.unsqueeze(0).unsqueeze(0)
            x0, sigma, eps, cond = generate_train_sample(x0, schedule, False)
            sigma = sigma.to("cuda")
            eps = eps.to("cuda")
            
            model.train()
            optimizer.zero_grad()
            loss = model.get_loss(x0, sigma, eps, cond=cond)
            loss_track.append(loss.item())
            yield SimpleNamespace(**locals()) 
            loss.backward()
            optimizer.step()
        
        lrscheduler.step(loss)
        

def call_tainning_loop(config=None):                                        

    procsavepath = 'PRJ_AUTO_NOISEPARAM/data/processed/ANMO'
    procsavepathfile = os.path.join(procsavepath, 'anmo_noise_data_normalized.pkl')

    if os.path.isfile(procsavepathfile) == 1:
        with open(procsavepathfile, 'rb') as f:
            trainset = pickle.load(f)
    else:
        print("no data loaded")
    trainsubset = Subset(trainset, range(len(trainset)))


    torch.cuda.empty_cache()
                                       
    dataloader = DataLoader(trainsubset, batch_size=5000, shuffle=False)

    model = Scaled(Unet)(in_dim=128, in_ch=1, out_ch=1, 
                            ch_mult=tuple(config['chn_multiples']['values']),           
                            num_res_blocks=config['num_res_blocks']['values'],          
                            ch=config['ch_res']['values'],                               
                            attn_resolutions=(14,)).to("cuda")
    schedule = ScheduleLogLinear(N=100, sigma_min=0.1, sigma_max=10)

    ema = EMA(model.parameters(), decay=0.999)
    losslist= [] 

    for ns in training_loop(dataloader, model.to("cuda"), schedule, epochs=config['epochs']['value'], 
                            lr=config['lr_rate']['values']):
        ns.pbar.set_description(f'Epoch {ns.epoch} | Loss={ns.loss.item():.5}')
        losslist.append(ns.loss.item())
        ema.update()
        

    ##Save the model
    torch.save(model.state_dict(), "PRJ_AUTO_NOISEPARAM/models/diffusion_unet_attn.pth")


if __name__ == '__main__':
    
    config = {
    'chn_multiples': {
        'values': [1, 1, 2]
        },
    'epochs': {
        'value': 15000
        },
    'ch_res': {
          'values': 64
        },
    'lr_rate': {
          'values': 7e-4
        },
    'num_res_blocks': {
          'values': 1
        }
    }

    call_tainning_loop(config)

