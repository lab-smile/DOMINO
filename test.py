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
from scipy.io import savemat

#load monai functions - 

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    #AddChanneld,
    EnsureChannelFirstd,
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
    ScaleIntensityd
    RandGaussianNoised
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
#from monai.networks.nets import UNet

from monai.data import (
    DataLoader,
    load_decathlon_datalist,
    decollate_batch,
    Dataset,
    pad_list_data_collate,
)

#-----------------------------------

#set up starting conditions:

start_time = time.time()
print_config()

# our CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/red/nvidia-ai/SkylarStolte/training_pairs_v5_bfc/", help="directory the dataset is in")
parser.add_argument("--batch_size_test", type=int, default=1, help="batch size testing data")
parser.add_argument("--num_gpu", type=int, default=1, help="number of gpus")
parser.add_argument("--N_classes", type=int, default=12, help="number of tissues classes")
parser.add_argument("--spatial_size", type=int, default=256, help="one patch dimension")
parser.add_argument("--model_load_name", type=str, default="unetr_v5_bfc.pth", help="model to load")
parser.add_argument("--dataparallel", type=str, default="True", help="did your model use multi-gpu")
parser.add_argument("--a_max_value", type=int, default=255, help="maximum image intensity")
parser.add_argument("--a_min_value", type=int, default=0, help="minimum image intensity")
parser.add_argument("--json_name", type=str, default="dataset", help="name of the file used to map data splits")
args = parser.parse_args()

split_JSON = args.json_name #"dataset_1.json"
datasets = args.data_dir + split_JSON

#-----------------------------------

#data transformations:

test_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        #AddChanneld(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear"),
        ),
        Orientationd(keys=["image"], axcodes="RAS"),
        #ScaleIntensityRanged(
        #    keys=["image"], a_min=args.a_min_value, a_max=args.a_max_value, b_min=0.0, b_max=1.0, clip=True
        #),
        ScaleIntensityd(keys=["image"]),
	#CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image"]),
    ]
)

#-----------------------------------

#set up data loaders

test_files = load_decathlon_datalist(datasets, True, "test")

test_ds = Dataset(
    data=test_files, transform=test_transforms, 
)
test_loader = DataLoader(
    test_ds, batch_size=args.batch_size_test, shuffle=False, num_workers=4, pin_memory=True, collate_fn=pad_list_data_collate,
)

#-----------------------------------

#set up gpu device and unetr model

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataparallel == "True":

  model = nn.DataParallel(
      UNETR(
      in_channels=1,
      out_channels=args.N_classes, #12 for all tissues
      img_size=(args.spatial_size, args.spatial_size, args.spatial_size),
      feature_size=16, 
      hidden_size=768,
      mlp_dim=3072,
      num_heads=12,
      #pos_embed="perceptron",
      norm_name="instance",
      res_block=True,
      dropout_rate=0.0,
  ), device_ids=[i for i in range(args.num_gpu)]).cuda()
  
elif args.dataparallel == "False":

  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  model = UNETR(
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
  ).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

#-----------------------------------

model.load_state_dict(torch.load(os.path.join(args.data_dir, args.model_load_name)))
model.eval()

# Extract model name without extension
ModelName = os.path.splitext(os.path.basename(args.model_load_name))[0]
# Build path
save_dir = os.path.join(args.data_dir, "TestResults", ModelName)
# Create directories if they don’t exist
os.makedirs(save_dir, exist_ok=True)

import nibabel as nib

case_num = len(test_ds)
for i in range(case_num):
    with torch.no_grad():
        img_name = test_ds[i]["image"].meta["filename_or_obj"].split("/")[-1]
        img = test_ds[i]["image"]
        test_inputs = torch.unsqueeze(img, 1).cuda()
        test_outputs = sliding_window_inference(
            test_inputs,
            (args.spatial_size, args.spatial_size, args.spatial_size),
            4,
            model,
            overlap=0.8
        )

    print("Saving image " + str(i))
    testimage = torch.argmax(test_outputs, dim=1).detach().cpu().numpy()[0]  # remove batch dim

    # Load reference image with nibabel to get full header + affine
    ref_nii = nib.load(img.meta["filename_or_obj"])
    ref_header = ref_nii.header.copy()
    ref_affine = ref_nii.affine

    # Create new NIfTI with same header + affine
    new_img = nib.Nifti1Image(testimage.astype(np.uint8), affine=ref_affine, header=ref_header)

    # Build save path
    filename, _ = os.path.splitext(img_name)
    savepath = os.path.join(save_dir, filename + ".nii.gz")

    nib.save(new_img, savepath)

#------------------------------------

#time since start

print("--- %s seconds ---" % (time.time() - start_time))
