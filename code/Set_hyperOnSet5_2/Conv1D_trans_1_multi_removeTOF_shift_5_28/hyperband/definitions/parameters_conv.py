from hyperband.definitions.common_defs import *
from hyperopt import hp
from sklearn.metrics import roc_auc_score as AUC, log_loss, accuracy_score as accuracy
import torch as th
from torch.utils import data
import numpy as np
from data_loader import data_manager as dm
from model import models as m
from utils import utils
from parameter import parameter_hyperband as params_hyperband
import random


## params for hyperparameter search are defined here: parameter/parameter_hyperband.py
space = params_hyperband.get_space()


def get_params():
    params = sample(space)
#    params['conv_layer_1_strides'] = 1
#    params['conv_layer_1_batchnorm'] = {'name': True}
    return handle_integers(params)




#######################################################################################################
#######################################################################################################

def print_params(params):
    pprint({k: v for k, v in params.items() if not k.startswith('layer_')})
    print('')


def try_params(n_iterations, params, args_fixed, data_train, data_validate, data_test):
    n_epochs = n_iterations #*5  # one iteration equals 5 epochs
    print("epochs:", n_epochs)
#    print_params(params)

    ######################################################################################################
    ######################################################################################################
    ## load data with Data Manager ##


    # print("Load data...")
    # data_train = dm.LoadData(args=args_fixed, params=params, csvpath=args_fixed.traincsv_path)
    # data_validate = dm.LoadData(args=args_fixed, params=params, csvpath=args_fixed.validatecsv_path)
    # print("Done!\n")

    # create data manager


    seed = 10
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)

    data_manager = dm.DataManager(args=args_fixed, params=params, data=data_train, train=True)
    data_manager_val = dm.DataManager(args=args_fixed, params=params, data=data_validate, train=False)
    data_manager_test = dm.DataManager(args=args_fixed, params=params, data=data_test, train=False)



#    data_manager = dm.DataManager(args=args_fixed, params=params, csvpath=args_fixed.traincsv_path, data=data_train,train=True)
#    data_manager_test = dm.DataManager(args=args_fixed, params=params, csvpath=args_fixed.testcsv_path)
#    data_manager_val = dm.DataManager(args=args_fixed, params=params, csvpath=args_fixed.traincsv_path, data=data_validate, train=False)
    # Parameters
    train_params = {'batch_size': params['batch_size'],
                    'shuffle': True,
                    'num_workers': args_fixed.nb_workers,
                    'pin_memory': False}
    test_params = {'batch_size': params['batch_size'],
                    'shuffle': False,
                    'num_workers': args_fixed.nb_workers,
                    'pin_memory': False}

    # data
    training_generator = data.DataLoader(data_manager, **train_params)
#    training_generator_test = data.DataLoader(data_manager_test, **test_params)
    training_generator_val = data.DataLoader(data_manager_val, **test_params)
    training_generator_test = data.DataLoader(data_manager_test, **test_params)


    ######################################################################################################
    ######################################################################################################


#    input_tensor_size = next(iter(training_generator))[0].size()  # returns (batchsize, n_channels, length of signal)
#    input_size = input_tensor_size[-1]
#    input_channels = input_tensor_size[1] if len(input_tensor_size) > 2 else 1


    ######################################################################################################
    ######################################################################################################
    ## Model ##

    image,_ , _ , _ ,_ ,_ = data_manager_val[0]
    model = m.ParametricConvFCModel(image,args_fixed,params)

    loss_value_mean_val = 1e10

    ## check if model is valid ##
    if model.error is True:
        ## model is unvalid ##
#        loss_value_mean_test = np.inf
        loss_value_mean_val = np.inf
    else :

        print_params(params)


        ## model is valid ##
        #################################################################################################33
        #################################################################################################33

        ## loss function ##
        loss_func = utils.loss_function()

        ## optimizer ##
        optimizer = utils.choose_optimizer(args=args_fixed,params=params,model=model)


        print(model)
        # TRAINING
        device = th.device("cuda:" + str(args_fixed.gpu_id) if th.cuda.is_available() else "cpu")
        print(device)
        model.to(device)

        #################################################################################################33
        #################################################################################################33

        epoch_start = 0
        training_counter = 0

        for epoch in range(epoch_start, int(n_epochs)):
            model = model.train()

            #### Train ####
            model, optimizer, loss_value_mean = utils.train_network(model,optimizer,loss_func,device,training_generator,params)
            print("Train mean loss: {}".format(loss_value_mean))


            ## VALIDATION ##
    #        _ , _ , _ , loss_value_mean_test = utils.test_network(model,loss_func,device,args_fixed,training_generator_test)
    #        print("Test mean loss: {}".format(loss_value_mean_test))
            _ , _  , loss_value_val = utils.test_network(model,loss_func,device,args_fixed,training_generator_val)
            _ , _  , loss_value_test = utils.test_network(model,loss_func,device,args_fixed,training_generator_test)
            loss_value_mean = (loss_value_val+loss_value_test)/2
            print("Validation mean loss: {}".format(loss_value_mean))

            if loss_value_mean_val>loss_value_mean:
                loss_value_mean_val = loss_value_mean



#    loss = (loss_value_mean_test+loss_value_mean_val)/2
    loss = loss_value_mean_val
#    print('Test mean loss: {}, Validation mean loss: {}, MEAN loss: {}'.format(loss_value_mean_test,loss_value_mean_val,loss))
    print('Validation mean loss: {}, MEAN loss: {}'.format(loss_value_mean_val,loss))

    return {'loss': loss}
