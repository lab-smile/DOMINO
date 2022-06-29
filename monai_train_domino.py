# coding=utf-8

#load packages:

#standard packages - 

import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import time
import argparse
import math
import pandas as pd
import torch.nn.functional as F

#load monai functions - 

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    SpatialPadd,
    RandGaussianNoised
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from monai.optimizers import WarmupCosineSchedule
#from monai.networks.nets import UNet

from monai.data import (
    DataLoader,
    load_decathlon_datalist,
    decollate_batch,
    Dataset,
    pad_list_data_collate,
)

#from DOMINO_loss import criterion

#-----------------------------------

#set up starting conditions:

start_time = time.time()
print_config()

# our CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/mnt/training_pairs_v5/", help="directory the dataset is in")
parser.add_argument("--batch_size_train", type=int, default=1, help="batch size training data")
parser.add_argument("--batch_size_validation", type=int, default=1, help="batch size validation data")
parser.add_argument("--num_gpu", type=int, default=3, help="number of gpus")
parser.add_argument("--N_classes", type=int, default=12, help="number of tissues classes")
parser.add_argument("--spatial_size", type=int, default=256, help="one patch dimension")
parser.add_argument("--model_save_name", type=str, default="unetr_v5_cos", help="model save name")
parser.add_argument("--a_max_value", type=int, default=255, help="maximum image intensity")
parser.add_argument("--a_min_value", type=int, default=0, help="minimum image intensity")
parser.add_argument("--max_iteration", type=int, default=25000, help="number of iterations")
parser.add_argument("--num_samples", type=int, default=1, help="number of iterations")
parser.add_argument("--csv_matrixpenalty", type=str, default="/mnt/UNETR_matrixpenalty_v5_2d.csv", help="matrix penalty csv")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

split_JSON = "dataset_1.json"
datasets = args.data_dir + split_JSON

#-----------------------------------

#special loss

matrix_vals = pd.read_csv(args.csv_matrixpenalty, header = None)#, header = 0, index_col = 0) 
matrix_vals = matrix_vals.to_numpy()

ENCODINGS = 3.0 * matrix_vals

#print(ENCODINGS)
#1 x 12 (12 x 12)

#plt.figure()
#plt.imshow(ENCODINGS)
#plt.show()

matrix_penalty = torch.from_numpy(ENCODINGS).to(device)
matrix_penalty = matrix_penalty.float()

Npixels = args.spatial_size**3#image size #patch_size**3=64x64x64
length_targets = args.batch_size_train*args.num_samples#batch size #sampling_rate * batch_size =160
total_batch = Npixels*length_targets#image size * batch size #sampling_rate * batch_size * Npixels
N_classes = args.N_classes#number_of_classes

class criterion(nn.Module):#outputs,labels_vector, matrix_penalty):
    
    def __init__(self,weight=None, size_average=True):
        super(criterion,self).__init__()
    
    def forward(self, outputs, targets, matrix_penalty=matrix_penalty, N_classes=N_classes, Npixels=Npixels, length_targets=length_targets, total_batch=total_batch):
        
        #currently I set them like this to do each data point rather than whole batch at once
        penalty_term = torch.zeros(0).cuda()#-1).cuda()
        entropy_term = torch.zeros(0).cuda()#-1).cuda()
        matrix_penalty = matrix_penalty.cuda()

        #iterate through all labels of batch 
        #soft_outputs = F.softmax(outputs, dim=1)
        #for i in range(len(outputs)):
        #    penalty = torch.mm(F.one_hot(targets[i:i+1], 10).float(),matrix_penalty)
        #    penalty_term[i] = torch.mm(penalty.float(),torch.transpose(soft_outputs[i:i+1, :], 0, 1))
        
        target_vector = torch.reshape(targets, (length_targets, Npixels)) # B * P
        target_vector = F.one_hot(target_vector.to(torch.int64), N_classes) #int8 instead of int64, #B * P * N
        
        output_vector = torch.reshape(outputs, (length_targets, N_classes, Npixels)) #B * N * P
        output_vector = torch.swapaxes(output_vector, 0, 1) # N * B * P

        for i in range(length_targets): #B
            for j in range(len(target_vector)): #P
                target_vectorr = torch.flatten(target_vector[i:i+1, j:j+1, :]).float()  #1 * 1 * N = N
                target_vectorr = torch.reshape(target_vectorr, (1,N_classes)) # 1 x N
                
                output_vectorr = torch.flatten(output_vector[:, i:i+1, j:j+1]).float()  # N * 1 * 1
                output_vectorr = torch.reshape(output_vectorr, (N_classes,1)) # N x 1
                output_vectorr = torch.exp(output_vectorr)/torch.sum(torch.exp(output_vector)) #softmax
                
                penalty = torch.mm(target_vectorr,matrix_penalty)   # (1 x N) * (N x N) = 1 x N
                #penalty_term[B*P] = torch.mm(penalty.float(),output_vector) # (1 x N) * (N x 1) = 1 x 1
                penalty_term = torch.cat((penalty_term,torch.mm(penalty.float(),output_vectorr)),-1)
        
        loss_diceCE = DiceCELoss(to_onehot_y=True, softmax=True)
        entropy_term = loss_diceCE(outputs,targets)
        beta = 1.
        total_loss = entropy_term + beta*(torch.sum(penalty_term)/total_batch)
        return total_loss#/batch_size

#------------------------------------

#data transformations:

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min_value,
            a_max=args.a_max_value, #my original data is in UINT8
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        #CropForegroundd(keys=["image", "label"], source_key="image"), #can crop data since taking patches that are less than full
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(args.spatial_size, args.spatial_size, args.spatial_size),
            pos=1,
            neg=1,
            num_samples=args.num_samples, #this number poses a limitation on training since inputs are batch size x num_samples x spatial_size
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.10,
        ),
        RandGaussianNoised(keys = "image", prob = .50, mean = 0, std = 0.1),
        ToTensord(keys=["image", "label"]),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=args.a_min_value, a_max=args.a_max_value, b_min=0.0, b_max=1.0, clip=True
        ),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)

#-----------------------------------

#set up data loaders

train_files = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")
train_ds = Dataset(
    data=train_files,
    transform=train_transforms,
)
train_loader = DataLoader(
    train_ds, batch_size=args.batch_size_train, shuffle=True, num_workers=4, pin_memory=True, collate_fn=pad_list_data_collate,
)
val_ds = Dataset(
    data=val_files, transform=val_transforms, 
)
val_loader = DataLoader(
    val_ds, batch_size=args.batch_size_validation, shuffle=False, num_workers=4, pin_memory=True, collate_fn=pad_list_data_collate,
)

#-----------------------------------

#set up gpu device and unetr model

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.DataParallel(
    UNETR(
    in_channels=1,
    out_channels=args.N_classes, #12 for all tissues
    img_size=(args.spatial_size, args.spatial_size, args.spatial_size),
    feature_size=16, 
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
), device_ids=[i for i in range(args.num_gpu)]).cuda()

#model = model.to(device)

loss_function = criterion()#DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

#-----------------------------------

def validation(epoch_iterator_val):
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (args.spatial_size, args.spatial_size, args.spatial_size), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
            )
        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)
    return mean_dice_val

#-----------------------------------

def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(args.data_dir, args.model_save_name + ".pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best

#-----------------------------------

max_iterations = args.max_iteration #25000
eval_num = math.ceil(args.max_iteration * 0.02)#500
#WarmupCosineSchedule(optimizer, 10, 500, cycles = 0.5)
post_label = AsDiscrete(to_onehot=True, num_classes=args.N_classes) 
post_pred = AsDiscrete(argmax=True, to_onehot=True, num_classes=args.N_classes) 
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )
model.load_state_dict(torch.load(os.path.join(args.data_dir, args.model_save_name + ".pth")))

#-----------------------------------

#training loss and validation evaluation

plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Iteration")
plt.plot(x, y)

dict = {'Iteration': x, 'Loss': y}  
df = pd.DataFrame(dict) 
df.to_csv(os.path.join(args.data_dir,args.model_save_name + '_Loss.csv'))

plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [eval_num * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("Iteration")
plt.plot(x, y)
#plt.show()
plt.savefig(os.path.join(args.data_dir, args.model_save_name + "_training_metrics.pdf"))

dict = {'Iteration': x, 'Dice': y}  
df = pd.DataFrame(dict) 
df.to_csv(os.path.join(args.data_dir,args.model_save_name + '_ValidationDice.csv'))

#------------------------------------

#time since start
print("--- %s seconds ---" % (time.time() - start_time))
