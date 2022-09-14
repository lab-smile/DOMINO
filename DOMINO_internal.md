## DOMINO Code Train & Test Tutorial Windows 10 Version

#### **Intro**
The purpose of this manual is to help you successfully run the DOMINO train and test code on the HiperGator through terminal commands and MATLAB GUI on Windows 10 operationg system, which contain the following parts:
- Get the code and the data you need;
- Introduction to the container;
- Run the code by Linux command;
- Check your result;

#### **Prerequisites**
All operations described in this manual are based on a comprehensive enhanced terminal, MobaXterm. You can also use the other cross-platform server application as your preference.

#### **Obatian DOMINO code and dataset**
In order not to disturb the others’ work environment, you need to go to your own directory by typing the command below, you can see the prefix changed when you change your directory:
```
cd /blue/ruogu.fang/[your username]
```
All of the code you need can be found in this GitHub repository:
```
https://github.com/lab-smile/DOMINO
```
The username here is your GitHub user name, the password here is the token you created by yourself on the GitHub. You can find a clear explanation of token in the “SMILE GPU Server Tutorial”.
<br>
And you can find the data under /blue/ruogu.fang/share/ACT_og_v5 on Hipergator.<br>
1. Run the MATLAB codes by MATLAB GUI
You can find there are two MATLAB codes, these two MATLAB codes should not require any changes to the path. To run these two MATLAB codes, you need to initiate MATLAB GUI by using Open OnDemand. You can open the Open OnDemand by following link:  
```
https://ood.rc.ufl.edu
```
You can find MATLAB under the Home/My interactive Sessions and setting the parameters as the following list:

- MATLAB version: choose the MATLAB version you are used to use. In the tutorial, the MATLAB version is R2020b;
- Number of CPU cores requested per MPI task: 1;
- Maximum memory requested for this job in Gigabytes: 10;
- SLURM Account/ Qos: Setting these to parameters as same as your HiperGator group account;
- Time Requested for this job in hours: this parameter is the time limit for this job. In general, running the two MATLAB codes in DOMINO should not take more than three hours. In case you need to make some change, I recommand require 48 hours for ood jobs;
- The other parameters you can keep as default value.

Then you can launch the MATLAB GUI by using Open OnDemand.

You need to select the DOMINO working folder and add to path/ Selected Folders and Subfolders firstlt before you running these two MATLAB code. Then you can directly run the combine_mask.m without making any change. But if you are using MATLAB R2020b, you need to change line 56 to 
```
image(index) = tissue_cond_updated.Labels(k)
```
 The output should be a Data folder with the following structure: 
 ```
 Data ImagesTr sub-TrX_T1.nii sub-TrXX_T1.nii ... 
 ImagesTs sub-TsX_T1.nii sub-TsXX_T1.nii ...
 LabelsTr sub-TrX_seg.nii sub-TrXX_seg.nii ...
 LabelsTs sub-TsX_seg.nii sub-TsX_seg.nii ...
 ```
 Then you need to change your directory to the data folder(/ACT_og_v5) and run makeGRACEjson.m. After this code done, you may exit MATLAB and open terminal to run the other codes.

2. Training and Testing the DOMINO code by using terminal
Firstly you need to cd to your own code directory. The DOMINO code is using the MONAI, which is an open source foundation. We need to build our own container for MONAI as you build a environment for your code. Run build_container_v08.sh by using the command: 
```
sbatch building_container_v08.sh
```
You need to change the mail-user to your own HPG mail address, change the line after sandbox to your desired writable directory and change the line nv to your own directory, I post my own changing as an example:![build_container_example.png](https://s2.loli.net/2022/07/05/7Xv3U8rH1fQcCPu.png)
Then you can find there is a folder named monaicore08 generated under your desired directory.<br>
Now you can train and test the DOMINO code. To train the DOMINO code, you need to run the runMONAI.sh. Before you training progress, you also need to change the mail-user and the directory. One tips here is that you may change the max_iteration to 100 to see if the code can train successfully. Here is my runMONAI.sh as an example: ![runMONAI.png](https://s2.loli.net/2022/09/14/uU1ILnXwqMBmKhC.png) For the iterations = 100, the training progress might take about one hours, and for the iterations = 25,000, the training progress might take about 24 hours. Here is another tips, do not forget change the model_save_name to avoid the new model overwritten the old model. Then you can find your training result in the monai_job number.log file which is generated after the training job done.The outputs of the trained model along with the training and validation metrics were recorded during training. 
The test progress is very similar to the training progress. You need to change all paths and make sure the model_save_name matches your model name in runMONAI.sh. Then running the runMONAI_test.sh with sbatch. ![runMONAI_test.png](https://s2.loli.net/2022/09/14/uU1ILnXwqMBmKhC.png)
The outputs for each test subject is saved as a mat file.