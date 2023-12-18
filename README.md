# Bone-Ablation Depth Estimation from Er-YAG Laser generated Acoustic Waves

This is the code to the paper [Bone-Ablation Depth Estimation from Er-YAG Laser generated Acoustic Waves](https://doi.org/10.1109/ACCESS.2022.3225651) (IEEE Access 2022).

The description of which transducer is used can be found in the folder transducer. We made measurements for the transducer, but for the paper, we used solely one transducer. The details can be found under the name [Model_WSa_trans1](transducer/Model_WSa_trans1.pdf).

The data was divided into 5 disjunct [subsets](code/bone_division.txt). Each folder in the code uses one of the divisions. 
To run the code enter one of these folders, e.g.
```
cd code/Set_hyperOnSet5_5
```
Here, we have five subfolders, which correspond to each network described in the paper. To get reproducible results you need to train the network and test the network for each subfolder, e.g.

```
CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=3 python3 Conv1D_trans_1_multi_removeTOF_shift_5/train_model.py --logfile
```
```
CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=3 python3 Conv1D_trans_1_multi_removeTOF_shift_5/test_model.py --infer-model best
```
This should be done for all the five folders in each subset of data. In the final step, the cross-validation is done with the file.

`compareNN5.m`

We note, that the linear approximation is done in

`linearApproximation.m`

If you need the data, open an issue. We will give you a temporary link to download the data. 
