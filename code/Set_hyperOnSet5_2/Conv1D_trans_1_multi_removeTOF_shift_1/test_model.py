import torch as th
import numpy as np
from data_loader import data_manager as dm
from torch.utils import data
from model import models as m
from utils import utils
from parameter import run_args
import pathlib

def infer_and_evaluate(args, state, chose_path):
    params = state['params']
    ## loss function
    loss = state['loss']

    ######################################################################################################
    ######################################################################################################
    ## load data with Data Manager ##
    print("\nLoad data...")
    if chose_path is 'train':
        interval = args.train_interval
    elif chose_path is 'validate':
        interval = args.validate_interval
    else:
        interval = args.test_interval

    data_train = dm.LoadData(args=args,interval=interval)
    print("Done!\n")

    data_manager_test = dm.DataManager(args=args, params=params, data=data_train, train=False) 
    params_test = {'batch_size': params['batch_size'],
                   'shuffle': False,
                   'num_workers': args.nb_workers,
                   'pin_memory': True}

    training_generator_test = data.DataLoader(data_manager_test, **params_test)

    ######################################################################################################
    ######################################################################################################
    ## Model ##

    image, _, _, _, _, _  = data_manager_test[0]
    model = m.ParametricConvFCModel(image, args, params)

    print(model)
    model.load_state_dict(state['network'])

    print("******************** CUDA ***********************")
    device = th.device("cuda:" + str(args.gpu_id) if th.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    ##########################################################################################################
    ##########################################################################################################
    #### Evaluate ####

#    output_var, _, output, loss_value_mean = utils.test_network(model, loss, device, args
#                                                                                     , training_generator_test, test=True)

    output, label,  loss_value_mean = utils.test_network(model, loss, device, args, training_generator_test)


    print("mean loss: {}".format(loss_value_mean))

    ##########################################################################################################
    ##########################################################################################################

    # weighted_mean_accuracy = 0
    # for i in range(len(args.classes)):
    #     weighted_mean_accuracy += 100 * class_correct["array"][i] / class_total["array"][i]
    # weighted_mean_accuracy = weighted_mean_accuracy / len(args.classes)
    #
    # print('************* TEST DATA *******************')
    # print('Accuracy of the network on the test images: %.5f %%' % (weighted_mean_accuracy))
    #
    # for i in range(len(args.classes)):
    #     print('Accuracy of %10s : %3.2f %%' % (
    #         args.classes[i], 100 * class_accuracy["array"][i]))
    # print("*******************************************************")
    # print(class_correct["matrix"].astype(int))



    # output = output*data_train.sigma+data_train.mean
#    output = output.numpy()
#    output_var = output_var.numpy()
    pathlib.Path(args.path_to_folder+'/results/gradcam').mkdir(parents=True, exist_ok=True)
    np.savetxt(args.path_to_folder + "/results/gradcam/output.txt", output.astype(float), fmt='%f')
    np.savetxt(args.path_to_folder + "/results/gradcam/label.txt", label.squeeze(1).astype(float), fmt='%f')
#    np.savetxt(args.path_to_folder + "/output_var.txt", output_var.astype(float), fmt='%d')
#    np.savetxt(args.path_to_folder + "/pos.txt", data_train.pos.astype(int), fmt='%d, %d')
#    np.savetxt(args.path_to_folder + "/path.txt", data_train.str, delimiter=" ", newline = "\n", fmt="%s")

    ##########################################################################################################
    ##########################################################################################################


if __name__ == "__main__":

    args = run_args.parser.parse_args()
    dataset_for_inference = args.infer_data

    if dataset_for_inference == 'train':
        chose_path = 'train'
#        chose_path = args.traincsv_path
    elif dataset_for_inference == 'test':
        chose_path = 'test'
#        chose_path = args.testcsv_path
    elif dataset_for_inference == 'validate':
        chose_path = 'validate'
#       chose_path = args.validatecsv_path
    else:
        raise NotImplementedError


    checkpoint_to_load = args.infer_model
    if checkpoint_to_load == 'last':
        path = args.path_to_folder+'/results/model/model_last.pt'
        state = th.load(path)
        args = state['args']
        mod_model = 'last'
    elif checkpoint_to_load == 'best':
        path = args.path_to_folder+'/results/model/model_best.pt'
        state = th.load(path)
        args = state['args']
        mod_model = 'best'
    else:
        raise NotImplementedError

    print("Infer and evaluate on {} data, using the {} checkpoint".format(dataset_for_inference, checkpoint_to_load))
    print("change data using the --infer_data flag to train, validate or test")
    print("change checkpoint using the --infer_model flag to last or best")

    epoch_start = state['epoch']
    print("****************************************************************")
    print("************* epoch ", epoch_start, ", ***********************")
    print("************* path ", chose_path, ", ***********************")
    print("****************************************************************")

    print(chose_path)
    infer_and_evaluate(args, state, chose_path)
