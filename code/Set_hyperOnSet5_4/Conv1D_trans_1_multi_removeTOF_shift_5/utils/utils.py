import os
import torch as th
import scipy.io
import numpy as np
import scipy.fftpack as ft
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
#from audtorch.metrics.functional import pearsonr
import torch.nn.functional as F
import math
# import tk
# import tkinter



#####################################################################################
#####################################################################################
def save_checkpoint(state, path, filename):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=False)
    th.save(state, os.path.join(path, filename))




class loss_function():

    MSELoss = th.nn.MSELoss(reduction='mean')
    # MSELoss = th.nn.L1Loss()
    # MSELoss =

    CrossEntropyLoss= th.nn.CrossEntropyLoss(reduction='mean')
    SmoothL1Loss = th.nn.SmoothL1Loss(beta=0.5) #params["beta"])

    # def __init__(self, output, label):
    #     self.loss_var = 0




#####################################################################################
#####################################################################################


## programm the traing part of the network ##
def train_network(model,optimizer, loss,device,training_generator,params):

      ## Output:
      #  model
      #  mean_loss_value



      model = model.train()

#      optimizer.zero_grad()
#      model.zero_grad()

      loss_value_mean = 0.
      train_counter = 0
      for spec, target_label, spec_small, target_label_small, spec_big, target_label_big in training_generator:


            if spec.shape[0]>1:

                optimizer.zero_grad()
                model.zero_grad()

                if spec.shape[0]>1:

                    if train_counter == 0:
                        batch_size = target_label.size(0)

                    input = spec.to(device)
                    depth = (target_label[:,:,1]-target_label[:,:,0]).to(device)

                    input_small = spec_small.to(device)
                    input_big = spec_big.to(device)


                    ################################################################################################
                    net_output, net_output_cnn = model(input)


                    loss_value = loss.MSELoss(net_output,depth)

                    loss_value.backward()
                    optimizer.step()

                    loss_value_mean += loss_value*input.shape[0]

                    ## print loss value ##
                    ## print only ~6 loss values
                    if train_counter % np.floor(len(training_generator)/5) == 0:
                       print("loss function at mini-batch iteration {} : {}".format(train_counter, loss_value))
                    train_counter += 1


      loss_value_mean /= train_counter
      return model, optimizer, loss_value_mean



## programm the traing part of the network ##
def test_network(model,loss,device,args,training_generator_test,test=False):

    ## Output:
    # class_correct     --  correct prediction
    # class_total       --  total true distribution
    # class_accuracy    --  accuracy
    #   - matrix        --  confucius matrix
    #   - array         --  represent it as array
    #   - single        --  all in an unweighted single value
    # loss_value_mean   --  mean loss



    with th.no_grad():
        model = model.eval()

        output = th.tensor([]);
        label  = th.tensor([])
        loss_value_mean = 0
        # cc = 0
        for spec, target_label, _ , _ , _ , _  in training_generator_test:
            # cc += 1
            # print(cc)
            input = spec.to(device)
            net_output, _ = model(input)

            output  = th.cat((output, net_output.to("cpu")), 0)
            label  = th.cat((label, (target_label).to("cpu")), 0)



        loss_value_mean  = loss.MSELoss(output,label[:,:,1]-label[:,:,0])

    return output.numpy(), label.numpy(),  loss_value_mean


####################################################################################
def removeTOF(data, input_size, scale=1.5, maxPosOfNoise=500):

    data_ = data[:,0:input_size].clone()

    for i in range(data.shape[0]):
        data_max_noise = max(abs(data[i,0:maxPosOfNoise]))
        oo = ((data_max_noise*scale)<abs(data[i,:])).int()
        oo_begin  = th.argmax(oo)
        data__ = data[i,oo_begin:(oo_begin+input_size)].clone()

        data__ = (data__-th.mean(data__))/th.sqrt(th.var(data__))
        data_[i,:] = data__

    return data_




#####################################################################################
#####################################################################################
## choose optimizer ##
def choose_optimizer(args,params,model):
      optimizer = None
      if params['optimizer'] == 'RMSprop':
          optimizer = th.optim.RMSprop(model.parameters(), lr=params['learnin_rate'],weight_decay=params['L2_reguralization'])
      elif params['optimizer'] == 'Adam':
          optimizer = th.optim.Adam(model.parameters(), lr=params['learnin_rate'], amsgrad=args.amsgrad,weight_decay=params['L2_reguralization']) #additional parameter? amsgrad?
      elif params['optimizer'] == 'Rprop':
          optimizer = th.optim.Rprop(model.parameters(), lr=params['learnin_rate'])
      elif params['optimizer'] == 'SGD':
          optimizer = th.optim.SGD(model.parameters(), lr=params['learnin_rate'],weight_decay=params['L2_reguralization'])

      return optimizer

def choose_optimizer_cnn(args,params,model):
      optimizer = None
      if params['optimizer_cnn'] == 'RMSprop':
          optimizer = th.optim.RMSprop(model.parameters(), lr=params['learnin_rate_cnn'],weight_decay=params['L2_reguralization_cnn'])
      elif params['optimizer_cnn'] == 'Adam':
          optimizer = th.optim.Adam(model.parameters(), lr=params['learnin_rate_cnn'], amsgrad=args.amsgrad,weight_decay=params['L2_reguralization_cnn']) #additional parameter? amsgrad?
      elif params['optimizer_cnn'] == 'Rprop':
          optimizer = th.optim.Rprop(model.parameters(), lr=params['learnin_rate_cnn'])
      elif params['optimizer_cnn'] == 'SGD':
          optimizer = th.optim.SGD(model.parameters(), lr=params['learnin_rate_cnn'],weight_decay=params['L2_reguralization_cnn'])
   
      return optimizer

