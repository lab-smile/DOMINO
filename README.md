# DOMINO: Domain-aware Model Calibration in Medical Image Segmentation
We address a model called a **domain-aware model calibration method**, nicknamed **DOMIMO** that leverages the semantic confusability and hierarchical similarity between class labels. Our experiments demonstrate that our DOMINO-calibrated deep neural networks outperform non-calibrated models and state-of-the-art morphometric methods in head image segmentation. Our results show that our method can consistently achieve better calibration, higher accuracy, and faster inference times than these methods, especially on rarer classes. This performance is attributed to our domain-aware regularization to inform semantic model calibration. These findings show the importance of semantic ties between class labels in building confidence in deep learning models. The framework has the potential to improve the trustworthiness and reliability of generic medical image segmentation models.

## Paper
This repository provides the official implemantation of training DOMINO as well as the usage the model DOMINO in the following paper:
https://arxiv.org/abs/2209.06077

## Major results from our work






## Usage
You can find there are two MATLAB codes, you can directly change the directory to your own data. You need to select the DOMINO working folder and add to path before you running these two MATLAB codes. 

To run the combineIn case of you are using different version of MATLAB, if you are using MATLAB 2020b, you need to change line 56 to :
```
image(index) = tissue_cond_updated.Labels(k)
```
Then you can run the combine_mask.m. The output should be a Data folder with the following structure: 
```
Data ImagesTr sub-TrX_T1.nii sub-TrXX_T1.nii ... 
ImagesTs sub-TsX_T1.nii sub-TsXX_T1.nii ...
LabelsTr sub-TrX_seg.nii sub-TrXX_seg.nii ...
LabelsTs sub-TsX_seg.nii sub-TsX_seg.nii ...
```
Maneuver to the /your_data/Data/. Run makeGRACEjson.m

After this code is done, you may exit MATLAB and open the terminal to run the other codes.

### Build container
The DOMINO code uses the MONAI, an open-source foundation. We provide a .sh script to help you to build your own container for running your code.

Run the following code in the terminal, you need tochange the line after --sandbox to your desired writable directory and change the line after --nv to your own directory.
```
sbatch building_container_v08.sh
```

The output should be a folder named monaicore08 under your desired directory.

### Training
Once the data and the container are ready, you can train the model by using the following command:
```
sbatch runMONAI.sh
```
Before you training the model, you need to make sure change the following directory:
- change the first singularity exec -nv to the directory includes monaicore08, for example: /user/DOMINO/monaicore08
- change the line after --bind to the directory includes monaicore08
- change the data_dir to your data directory
- change the model name to your desired model name
You can also specify the max iteration number for training. For the iterations = 100, the training progress might take about one hours, and for the iterations = 25,000, the training progress might take about 24 hours. 

### Testing
The test progress is very similar to the training progress. You need to change all paths and make sure the model_save_name matches your model name in runMONAI.sh. Then running the runMONAI_test.sh with sbatch command:
```
sbatch runMONAI_test.sh
```

## Citation
If you use this code, please cite our papers:
```@InProceedings{zhou2019models,
  author="Stolte, Skylar E. and Volle, Kyle and Indahlastari, Aprinda and Albizu, Alejandro and Woods, Adam J. and Brink, Kevin and Hale, Matthew and Fang, Ruogu",
  title="DOMINO: Domain-aware Model Calibration in Medical Image Segmentation",
  booktitle="Medical Image Computing and Computer Assisted Intervention (MICCAI) 2022 Oral Talk",
  year="2022",
  url="https://arxiv.org/abs/2209.06077"
}
```
## Acknowledgement
We employ UNETR as our base model from:
https://github.com/Project-MONAI/research-contributions/tree/main/UNETR
```
@InProceedings{hatamizadeh2022unetr,
  title={Unetr: Transformers for 3d medical image segmentation},
  author={Hatamizadeh, Ali and Tang, Yucheng and Nath, Vishwesh and Yang, Dong and Myronenko, Andriy and Landman, Bennett and Roth, Holger R and Xu, Daguang},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={574--584},
  year={2022}
}
```
## Contact
Any discussion, suggestions and questions please contact: [Skylar Stolte](mailto:skylastolte4444@ufl.edu), [Dr. Ruogu Fang](mailto:ruogu.fang@bme.ufl.edu).

*Smart Medical Informatics Learning & Evaluation Laboratory, Dept. of Biomedical Engineering, University of Florida*
