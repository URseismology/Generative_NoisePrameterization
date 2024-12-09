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
import wandb
import pprint
import sys
sys.path.insert(0, '/home/urseismoadmin/Documents')
from PRJ_AUTO_NOISEPARAM.code.lib.diff_model_utils import *
from PRJ_AUTO_NOISEPARAM.code.lib.utils import *


def training_loop(loader  : DataLoader,
                  model   : nn.Module,
                  schedule: Schedule,
                  epochs  : int = 10000,
                  lr      : float = 1e-3):
    wandb.watch(model,nn.MSELoss(),log="all",log_freq=100)                                      ##uncomment for using with wandb
    
    optimizer = optim.AdamW(model.parameters(),lr=lr)
    lrscheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=1300)

    loss_track = []
    model.train()
    #for epoch in range(epochs):
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
        wandb.log({"epoch":epoch, "loss":loss}, step=epoch)                                      ##uncomment for using with wandb
        

def call_tainning_loop(config=None):
    wandb.login()                                                                                ##uncomment for using with wandb

    #print(os.path.dirname(os.path.realpath(__file__)))
        
    procsavepath = '/home/urseismoadmin/Documents/PRJ_AUTO_NOISEPARAM/data/processed/ANMO'
    procsavepathfile = os.path.join(procsavepath, 'anmo_noise_data_normalized.pkl')

    if os.path.isfile(procsavepathfile) == 1:
        with open(procsavepathfile, 'rb') as f:
            trainset = pickle.load(f)
    else:
        print("no data loaded")
    trainsubset = Subset(trainset, range(len(trainset)))



    torch.cuda.empty_cache()
    with wandb.init(config=config):                                               ##uncomment for using with wandb
        config = wandb.config
        dataloader = DataLoader(trainsubset, batch_size=5000, shuffle=False)

        model = Scaled(Unet)(in_dim=128, in_ch=1, out_ch=1, 
                             ch_mult=tuple(config.chn_multiples),         #(2, 4, 8), 
                             num_res_blocks=config.num_res_blocks, #1, 
                             ch=config.ch_res,                     #32, 
                             attn_resolutions=(14,)).to("cuda")
        schedule = ScheduleLogLinear(N=100, sigma_min=0.1, sigma_max=10)

        ema = EMA(model.parameters(), decay=0.999)
        losslist= [] 


        for ns in training_loop(dataloader, model.to("cuda"), schedule, epochs=config.epochs, lr=config.lr_rate):
            ns.pbar.set_description(f'Epoch {ns.epoch} | Loss={ns.loss.item():.5}')
            losslist.append(ns.loss.item())
            ema.update()
        


if __name__ == '__main__':

    print(f"__name__ = {__name__}")
    sweep_config = {
        'method': 'random'
    }
    metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }
    sweep_config['metric'] = metric
    
    parameters_dict = {
    'chn_multiples': {
        'values': [[2, 4, 8],[1, 1, 2],[1, 2, 2]]
        },
    'epochs': {
        'value': 15000
        },
    'ch_res': {
          'values': [32, 64]
        },
    'lr_rate': {
          'values': [7e-4, 5e-4, 1e-4, 7e-5, 5e-5, 1e-5]
        },
    'num_res_blocks': {
          'values': [1,2,3]
        }
    }
    sweep_config['parameters'] = parameters_dict
    pprint.pprint(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="seis-noise-diff-covar-learn-unet-main-sweep-v1")
    wandb.agent(sweep_id, call_tainning_loop, count=10)


