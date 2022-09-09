import pathlib
import sys
import time

import numpy as np
import torch as th
from torch.utils import data

import hyperband.definitions.common_defs as cc
from data_loader import data_manager as dm
from model import models as m
from parameter import run_args
from utils import utils
import random

def print_params(params):
    cc.pprint({k: v for k, v in params.items() if not k.startswith('layer_')})
    print('')


def train(args, epoch_start, state):
    if epoch_start > 0:
        params = state['params']
    else:
        params =  {'L2_reguralization': 0,
 'activation': 'relu',
 'alpha': 0.4,
 'amplitude_shift': 0.5,
 'batch_size': 32,
 'conv_layer_1_batchnorm': {'name': True},
 'conv_layer_1_channels_out': 32,
 'conv_layer_1_extras': 'name',
 'conv_layer_1_kernel_size': 4,
 'conv_layer_1_kernel_size_maxpool': 3,
 'conv_layer_1_strides': 1,
 'conv_layer_1_strides_maxpool': 2,
 'conv_layer_2_batchnorm': {'name': False},
 'conv_layer_2_channels_out': 64,
 'conv_layer_2_extras': 'name',
 'conv_layer_2_kernel_size': 4,
 'conv_layer_2_kernel_size_maxpool': 3,
 'conv_layer_2_strides': 2,
 'conv_layer_2_strides_maxpool': 2,
 'conv_layer_3_batchnorm': {'name': False},
 'conv_layer_3_channels_out': 8,
 'conv_layer_3_extras': 'name',
 'conv_layer_3_kernel_size': 4,
 'conv_layer_3_kernel_size_maxpool': 2,
 'conv_layer_3_strides': 1,
 'conv_layer_3_strides_maxpool': 1,
 'conv_layer_4_batchnorm': {'name': False},
 'conv_layer_4_channels_out': 256,
 'conv_layer_4_extras': 'name',
 'conv_layer_4_kernel_size': 5,
 'conv_layer_4_kernel_size_maxpool': 3,
 'conv_layer_4_strides': 2,
 'conv_layer_4_strides_maxpool': 2,
 'conv_layer_5_batchnorm': {'name': True},
 'conv_layer_5_channels_out': 256,
 'conv_layer_5_extras': 'name',
 'conv_layer_5_kernel_size': 4,
 'conv_layer_5_kernel_size_maxpool': 2,
 'conv_layer_5_strides': 2,
 'conv_layer_5_strides_maxpool': 1,
 'conv_layer_6_batchnorm': {'name': False},
 'conv_layer_6_channels_out': 256,
 'conv_layer_6_extras': 'name',
 'conv_layer_6_kernel_size': 3,
 'conv_layer_6_kernel_size_maxpool': 3,
 'conv_layer_6_strides': 2,
 'conv_layer_6_strides_maxpool': 1,
 'conv_layer_7_batchnorm': {'name': False},
 'conv_layer_7_channels_out': 8,
 'conv_layer_7_extras': 'name',
 'conv_layer_7_kernel_size': 4,
 'conv_layer_7_kernel_size_maxpool': 2,
 'conv_layer_7_strides': 1,
 'conv_layer_7_strides_maxpool': 1,
 'conv_layer_8_batchnorm': {'name': True},
 'conv_layer_8_channels_out': 128,
 'conv_layer_8_extras': 'name',
 'conv_layer_8_kernel_size': 2,
 'conv_layer_8_kernel_size_maxpool': 3,
 'conv_layer_8_strides': 2,
 'conv_layer_8_strides_maxpool': 1,
 'conv_layer_9_batchnorm': {'name': False},
 'conv_layer_9_channels_out': 4,
 'conv_layer_9_extras': 'name',
 'conv_layer_9_kernel_size': 3,
 'conv_layer_9_kernel_size_maxpool': 3,
 'conv_layer_9_strides': 2,
 'conv_layer_9_strides_maxpool': 2,
 'conv_order': True,
 'depthShift': 2.6,
 'depthShift_sigma': 0.8,
 'fc_layer_1_batchnorm': {'name': True},
 'fc_layer_1_extras': {'name': 'dropout', 'rate': 0.4853609126406153},
 'fc_layer_2_batchnorm': {'name': True},
 'fc_layer_2_extras': {'name': None},
 'fc_layer_3_batchnorm': {'name': True},
 'fc_layer_3_extras': {'name': None},
 'fc_layer_4_batchnorm': {'name': True},
 'fc_layer_4_extras': {'name': 'dropout', 'rate': 0.5430694454362267},
 'fc_layer_5_batchnorm': {'name': False},
 'fc_layer_5_extras': {'name': 'dropout', 'rate': 0.4345536297888637},
 'fc_layer_6_batchnorm': {'name': True},
 'fc_layer_6_extras': {'name': 'dropout', 'rate': 0.5623788320304374},
 'fc_layer_7_batchnorm': {'name': True},
 'fc_layer_7_extras': {'name': 'dropout', 'rate': 0.1990679240917643},
 'fc_layer_8_batchnorm': {'name': True},
 'fc_layer_8_extras': {'name': 'dropout', 'rate': 0.20916768471525574},
 'fc_layer_9_batchnorm': {'name': True},
 'fc_layer_9_extras': {'name': None},
 'init': 'standard',
 'input_size': 2000,
 'learnin_rate': 0.001,
 'margin': 0.1,
 'n_conv_layers': 5,
 'n_fc_layers': 8,
 'neurons': 1028,
 'numberOfShots': 4,
 'optimizer': 'Adam'}



    seed = 3407
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    th.use_deterministic_algorithms(True)




    print_params(params)
    print(args.maxepoch)
    ######################################################################################################
    ######################################################################################################
    ## load data with Data Manager ##
    print("Load data...")


    data_train = dm.LoadData(args=args,interval=args.train_interval)
    data_validate = dm.LoadData(args=args,interval=args.validate_interval)
#    data_train = dm.LoadData(args=args,interval=[0,100])
#    data_validate = dm.LoadData(args=args,interval=[100,200])

    data_manager = dm.DataManager(args=args, params=params, data=data_train, train=True)
    data_manager_validate = dm.DataManager(args=args, params=params, data=data_validate, train=False)


    print("Done!\n")


    # Parameters
    train_params = {'batch_size': params['batch_size'],
                    'shuffle': True,
                    'num_workers': args.nb_workers,
                    'pin_memory': False}
#                    'pin_memory': True}
    validate_params = {'batch_size': params['batch_size'],
                       'shuffle': False,
                       'num_workers': args.nb_workers,
                       'pin_memory': False}
#                       'pin_memory': True}

    # data
    seed = 10
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)

    training_generator = data.DataLoader(data_manager, **train_params)
    training_generator_validate = data.DataLoader(data_manager_validate, **validate_params)

    ######################################################################################################
    ######################################################################################################
    ## Model ##

    device = th.device("cuda:" + str(args.gpu_id) if th.cuda.is_available() else "cpu")
    image, _, _, _, _, _ = data_manager_validate[0]
    model = m.ParametricConvFCModel(image, args, params)

    #################################################################################################33
    #################################################################################################33
    ## Check if model is valid"

    print(device)
    print(model)

    if model.error is True:
        print("Error: Change parameters!")
        print("*************************************")
    else:

        #################################################################################################
        #################################################################################################
        ## load network and initilize parameters ##
        if epoch_start > 0:
            ## Use old Network: load parameters and weights of network ##
            model.load_state_dict(state['network'])
            best_mean_loss = state['best_mean_loss']
            loss = state['loss']
        else:
            ## Use New Network: initialize parameters of network ##
            best_mean_loss = 1e10
            # loss function
            loss = utils.loss_function()

        ## optimizer ##
        optimizer = utils.choose_optimizer(args=args, params=params, model=model)
        #################################################################################################33
        #################################################################################################33

        model.to(device)
        train_counter = 0
        decay = 1
        for epoch in range(epoch_start, args.maxepoch):


#            if (epoch+1) % 20 is 0:
#                decay += 1
#                params["learnin_rate"] = params["learnin_rate"]/10 #decay
#                optimizer = utils.choose_optimizer(args=args, params=params, model=model)




            print("****************************************************")
            print("************* epoch ", epoch, "***************************")
            print("****************************************************")
            #### Train ####
            model, optimizer, loss_value_mean = utils.train_network(model, optimizer, loss, device, training_generator,params)
            print("mean loss: {}".format(loss_value_mean))

            #### Validate after each epoch ####
            print("***************************")
            print("********* Validate ************")
            print("***************************")
            class_correct, class_total, loss_value_mean_validate = utils.test_network(model, loss,
                                                                                                      device,
                                                                                                      args,
                                                                                                      training_generator_validate)
            print("mean loss: {}".format(loss_value_mean_validate))

            ##################################################################################
            ##################################################################################

            ## Do a weighted accuracy -- makes sense, if distribution of classes is unequal ##
            # weighted_mean_accuracy = 0
            # for i in range(len(args.classes)):
            #     weighted_mean_accuracy += 100 * class_correct["array"][i] / class_total["array"][i]
            # weighted_mean_accuracy = weighted_mean_accuracy / len(args.classes)

            ## save and print ##
            # print('************* Validate DATA *******************')
            # print('Accuracy of the network on the Validation images: %.5f %%' % (weighted_mean_accuracy))
            # np.savetxt(args.path_to_folder + "/results/model/class_predict_last", class_correct["matrix"].astype(int), fmt='%d')

            ## print confusion matrix ##
            # for i in range(len(args.classes)):
            #     print('Accuracy of %5s : %.5f %%' % (
            #         args.classes[i], 100 * class_accuracy["array"][i]))

            ##################################################################################
            ##################################################################################

            ## save last and best state ##

            ## we use mean_accuracy as measurment ##
            mean_loss_overall = loss_value_mean_validate

            state = {
                'train_counter': train_counter,
                'args': args,
                'network': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_mean_loss': best_mean_loss,
                'params': params,
                'loss': loss,
                }

            if best_mean_loss > mean_loss_overall:
                best_mean_loss = mean_loss_overall
                state['best_mean_loss'] = best_mean_loss
                utils.save_checkpoint(state, args.path_to_folder+'/results/model', "model_best.pt")
                # np.savetxt(args.path_to_folder+'/results/model' + "/class_predict_best",
                #            class_correct["matrix"].astype(int), fmt='%d')

            utils.save_checkpoint(state, args.path_to_folder+'results/model', "model_last.pt")

#            if epoch % 10 is 9:
#                utils.save_checkpoint(state, args.path_to_folder+'results/model', "model_"+str(epoch)+".pt")

            ##################################################################################
            ##################################################################################


if __name__ == "__main__":

    input_args = run_args.parser.parse_args()

    if input_args.mode == 'continue':
        print("Continue Training")
        path = args.path_to_folder+'/results/model/model_last.pt'
        state = th.load(path)
        args = state['args']
        epoch_start = state['epoch'] + 1
    elif input_args.mode == 'restart':
        print("Restart Training")
        args = input_args
        epoch_start = 0
        state = 0
    else:
        raise NotImplementedError('unknown mode')

    ## create folder, if they don't exist ##
    pathlib.Path(args.path_to_folder+'/results/output').mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.path_to_folder+'/results/model').mkdir(parents=True, exist_ok=True)

    ## write all the output in log file ##
    if input_args.logfile is True:
        time_string = time.strftime("%Y%m%d-%H%M%S")
        log_file_path = args.path_to_folder+'/results/output/output_{}.log'.format(time_string)
        print("Check log file in: ")
        print(log_file_path)
        sys.stdout = open(log_file_path, 'w')
    else:
        print("No log file, only print to console")

    train(args, epoch_start, state)
