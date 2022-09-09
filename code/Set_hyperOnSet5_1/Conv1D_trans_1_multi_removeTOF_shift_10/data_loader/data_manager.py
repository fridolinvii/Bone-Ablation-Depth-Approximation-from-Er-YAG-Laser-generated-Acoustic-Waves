import os
import torch as th
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from utils import utils

from torch.utils import data
import numpy as np
import random

from skimage import io, transform

#####################################################################################################
#####################################################################################################

class DataManager(data.Dataset):
    def __init__(self, args, params, data,train=True):


        self._data = data
        self._train = train
        self.params = params
        self.args = args


        print("Number of spectra in the data set", len(self._data.label))

    def sequence_length(self):
        return len(self._data.label)

    def __len__(self):
        return len(self._data.label)

    def __getitem__(self, idx):

        ## load data ##
        # self._labels_frame.iloc[idx, j], row  idx, column j of csv file
        # make sure you kow what written in the csv, and create input and label accordingly

        # Here,is Output:
        #                 - Image
        #                 - Label
        pixelSize = 10.86e-3
        label = self._data.label[idx]
        label = label*pixelSize
        label = label.unsqueeze(0)

        data = self._data.data[idx];

        # create multiple shots
        idx_0 = idx
        for n in range(self.params["numberOfShots"]):
            idx_n = idx-n
            label_n = self._data.label[idx_n]
            label_n = label_n*pixelSize
            label_n = label_n.unsqueeze(0)

            if label_n[0,0] == label[0,0]:  # still same shot
                idx_0 = idx_n

            data_n = self._data.data[idx_0]

            data = th.cat((data,data_n),dim=-1)


        ## Find similar ##
        # class 1: depth < 2.5mm; class 2: depth >= 2.5 mm


        if self._train:
            has_label_small = False
            has_label_big  = False

            if label[0,1]-label[0,0]<2.5:
                label_small = label
                label_big = label
                data_big = data
                data_small = data
                has_label_small = True
                has_label_big  = True
            else:
                while True:
                    idx_rand = th.randint(0,len(self._data.label),(1,))
                    label_ = self._data.label[idx_rand]*pixelSize
                    if label[0,1]-label[0,0]<2.5:
                        label_small = label
                        label_big = label
                        data_big = data
                        data_small = data
                        has_label_small = True
                        has_label_big  = True

                    elif label_[1]-label_[0] >= 2.5 and label_[1]-label_[0]<3:  # 2.5<=depth<3 
                        if has_label_small is False:
                            label_small = label_.unsqueeze(0)
                            has_label_small = True
                            data_small = self._data.data[idx_rand]
                            for n in range(self.params["numberOfShots"]):
                                idx_n = idx_rand-n
                                label_n = self._data.label[idx_n]
                                label_n = label_n*pixelSize
                                label_n = label_n.unsqueeze(0)

                                if label_n[0,0] == label[0,0]:  # still same shot
                                    idx_0 = idx_n

                                data_n = self._data.data[idx_0]
                                data_small = th.cat((data_small,data_n),dim=-1)
                    else: # label>=3mm
                        if has_label_big is False:
                            label_big = label_.unsqueeze(0)
                            has_label_big = True
                            data_big = self._data.data[idx_rand]
                            for n in range(self.params["numberOfShots"]):
                                idx_n = idx_rand-n
                                label_n = self._data.label[idx_n]
                                label_n = label_n*pixelSize
                                label_n = label_n.unsqueeze(0)

                                if label_n[0,0] == label[0,0]:  # still same shot
                                    idx_0 = idx_n

                                data_n = self._data.data[idx_0]
                                data_big = th.cat((data_big,data_n),dim=-1)
                    if has_label_small and has_label_big:
                        break
        else:
            label_small = 0
            label_big = 0
            data_big = 0
            data_small = 0

#        data_first = self._data.data_first[idx];




        if self._train:
#            time_shift = th.randint(0,self.params["time_shift"]+1,(3,))
            time_shift = [0,0,0]
            amplitude_shift = (2*th.rand(3)-1)*self.params["amplitude_shift"]
            data = th.transpose(data[time_shift[0]:-1,:],0,1)
            data_small = th.transpose(data_small[time_shift[1]:-1,:],0,1)
            data_big = th.transpose(data_big[time_shift[2]:-1,:],0,1)
#            data_first = th.transpose(data_first[(500+time_shift):(self.params["input_size"]+time_shift),self.args.transducer],0,1)

            data = data*(1+amplitude_shift[0]*th.exp(th.abs(data)))
            data_small = data_small*(1+amplitude_shift[1]*th.exp(th.abs(data_small)))
            data_big = data_big*(1+amplitude_shift[2]*th.exp(th.abs(data_big)))
#            data_first = data_first*(1+amplitude_shift*th.exp(th.abs(data_first)))


            data_small = utils.removeTOF(data_small,self.params["input_size"])
            data_big = utils.removeTOF(data_big,self.params["input_size"])

        else:
            data = th.transpose(data,0,1)
#            data_first = th.transpose(data_first[500:self.params["input_size"],self.args.transducer],0,1)

#        data = th.cat([data,data_first],dim=0)

        data = utils.removeTOF(data,self.params["input_size"])

        return data, label, data_small, label_small, data_big, label_big


## Preaload all the data
class LoadData:
    def __init__(self, args, interval):

        csv = pd.read_csv(os.path.join(args.path_to_folder,args.data_path))


        self.data        = []
        self.data_first  = []
        self.label       = []
        self.label_first = []




        for ii in range(interval[0],interval[1]):

            i = ii % len(csv)

            # load data
            image = pd.read_csv("data/"+csv.iloc[i,0],delimiter='\t')
            data = th.tensor(image.iloc[0:10000,1::].values,dtype=th.float32)
            data = data[:,args.transducer]
            self.data.append(data)

            # load label
            pos   =  [int(csv.iloc[i,1]),int(csv.iloc[i,2])]
            pos = th.tensor(pos,dtype=th.float32)
            self.label.append(pos)

            # first shot
            str = list(csv.iloc[i,0])
            str[-7:-4] = list('001')
            str = ''.join(str)
            image = pd.read_csv("data/"+str,delimiter='\t')
            data = th.tensor(image.iloc[0:8000,1::].values,dtype=th.float32)
            self.data_first.append(data) 









