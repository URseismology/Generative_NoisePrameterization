import numpy as np
import pandas as pd
import glob
import os

import obspy
from obspy import read,read_inventory,read_events

import torch
import pickle

class process_seismic_data(torch.utils.data.Dataset):
    def __init__(self,datapath='PRJ_AUTO_NOISEPARAM/data/raw/station',savepath='PRJ_AUTO_NOISEPARAM/data/processed/'):
        anmo_noise = []
        for file in glob.glob(datapath):
            if file.endswith('.Z'):
                st = read(file)
                tr = st.traces
                
                #getting metadata of the signals
                sample_rate = tr[0].stats.sampling_rate
                numberofsamples = tr[0].stats.npts
                delta_time = tr[0].stats.delta
                total_time_sec = numberofsamples*delta_time
                P_time = tr[0].stats.sac.t0
                data = st.traces[0].data  
                
                #performing de-mean and de-trend to remove any trends in the data and make the noise data zero-centered
                st.detrend(type='demean')
                st.detrend(type='linear')
                data_detrend_demean = st.traces[0].data  
                data_detrend_demean = (data_detrend_demean - data_detrend_demean.mean()) / data_detrend_demean.std() #normalizing step

                
                #Need to have all signals follow the same frequency of 20
                data_detrend_demean_antialias = None
                if(numberofsamples == 36000):
                    st.filter('lowpass', freq=0.4*20, zerophase=True) # anti-alias filter i.e. little less than the NYquist freq of desired freq.s
                    st.decimate(factor=int(40/20), no_filter=True)    # downsample
                    data_detrend_demean_antialias = st.traces[0].data
                    data_detrend_demean_antialias = (data_detrend_demean_antialias - data_detrend_demean_antialias.mean()) / data_detrend_demean_antialias.std() #normalizing step

                if(numberofsamples == 36000):
                    anmo_noise.append(data_detrend_demean_antialias[0:5000])
                elif (numberofsamples == 18000):
                    anmo_noise.append(data_detrend_demean[0:5000])
        
        traindata = np.array(anmo_noise)
        self.data = torch.from_numpy(traindata)
        self.length = traindata.shape[0]

    def __getitem__(self,index):
            return self.data[index]
            
    def __len__(self):
        return self.length

##example to execute the process and save file 
#pickleload_traindata('PRJ_AUTO_NOISEPARAM/data/raw/ANMO/*','PRJ_AUTO_NOISEPARAM/data/processed/')




