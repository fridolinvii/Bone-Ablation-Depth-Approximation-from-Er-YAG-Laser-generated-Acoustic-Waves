# Bone-Ablation Depth Approximation from Er-YAG Laser generated Acoustic Waves

The description of which transducer is used can be found in the folder transducer. We made measurment with for transducer, but for the paper we solely used one transducer. The details can be find under the name [Model_WSa_trans1](transducer/Model_WSa_trans1.pdf).

The data was divided in 5 disjunct [subsets](code/bone_division.txt). Each folder in code, uses one of the divisions. 
To run the code enter in one of these folders, e.g.
`cd code/Set_hyperOnSet5_5`

Here, we have five subfolders, which corrresponds to each network described in the paper. To get reproducible results you need to train the network and test the network for each subfolder, e.g.
`CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=3 python3 Conv1D_trans_1_multi_removeTOF_shift_5/train_model.py --logfile`
`CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=3 python3 Conv1D_trans_1_multi_removeTOF_shift_5/test_model.py --infer-model best`

This should be done for all the five folder in each subset of data. In a final step, the crossvalidation is done with the file
`compareNN5.m`

We note, that the linear approximation is done in
`linearApproximation.m`

If you need the data, ope an issue. We will give you a temporary link to download the data. 
