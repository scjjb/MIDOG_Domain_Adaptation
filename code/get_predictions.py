import random
import json
import math
import os
import numpy as np
from torchvision import models, transforms
from tqdm import tqdm_notebook as tqdm
from .segmentation_processing import preprocess_crops, get_model_predictions, add_bbox_to_map
from .neural_style_transfer_modelling import create_data_loaders_nst
from .segmentation_modelling import create_data_loaders, F1_score_centered



def get_default_prediction(model,model_type,train_scanners,valid_scanners=None,image_transforms=None,train_transforms=None,anchors=None,
                           threshold=0.5,min_detection_dim=10,evaldomain="train",evalset="train",device='cuda',max_patches=1000):
  assert model_type in ["unet","retinanet"], "model_type must be unet or retinanet"
  
  if valid_scanners==None:
    valid_scanners=train_scanners
    
  if train_transforms==None:
    train_transforms = transforms.Compose([
    ])
    
  if image_transforms==None:
    image_transforms = transforms.Compose([
      transforms.ToTensor(),
    ])
    
  
  if evaldomain=='train':
    eval_scanners=train_scanners
  elif evaldomain=='test':
    eval_scanners=valid_scanners

  if evalset=='train':
    _,image_ids,_ = preprocess_crops(10,max_patches,10,train_scanners=train_scanners,valid_scanners=eval_scanners)
  elif evalset=='test':
    _,_,image_ids = preprocess_crops(10,10,max_patches,train_scanners=train_scanners,valid_scanners=eval_scanners)
  elif evalset=='both':
    _,image_ids,test_ids = preprocess_crops(10,max_patches,max_patches,train_scanners=train_scanners,valid_scanners=valid_scanners)
    image_ids.extend(test_ids)
    random.shuffle(image_ids)
    image_ids=image_ids[:max_patches]
    
  print("total evaluation patches: ",len(image_ids))

  bboxs=[]
  true_bboxs=[]
  for a in range(math.ceil(len(image_ids)/500)):
    _,image_loader,_ = create_data_loaders(image_ids[:10],image_ids[a*500:(a+1)*500],test_ids=image_ids[:10],image_transform=image_transforms,train_transform=train_transforms)
    preds,masks,images=get_model_predictions(model,model_type,image_loader,device,max_batches=125,anchors=anchors)

    if model_type=="unet":
      bboxs.extend(add_bbox_to_map(preds,threshold=threshold,min_dim=min_detection_dim,return_images=False))
    elif model_type=="retinanet":
      bboxs.extend(preds)

    true_bboxs.extend(add_bbox_to_map(masks,threshold=threshold,return_images=False))

  return bboxs,true_bboxs



def get_mask_name_cgan(image_name):
  adjusted_image_name="_".join(image_name.split(".")[0].split("_")[:3])
  return adjusted_image_name[0:3]+"mask"+adjusted_image_name[3:]+".tiff" 

def image_end_path_cgan(image_name):
  return image_name[:3]+"/"+image_name


def get_cyclegan_prediction(model,model_type,train_scanners,valid_scanners,CGAN_model_name,CGAN_test_name,image_transforms=None,train_transforms=None,anchors=None,
                            threshold=0.5,min_detection_dim=10,results_dir="/drive/MyDrive/MIDOG_Style_Transfer/pytorch-CycleGAN-and-pix2pix/results/",
                            mask_name_func=get_mask_name_cgan,image_name_func=image_end_path_cgan,device='cuda',max_patches=1000):
  assert model_type in ["unet","retinanet"], "model_type must be unet or retinanet"

  if train_transforms==None:
    train_transforms = transforms.Compose([
  ])
    
  if image_transforms==None:
    image_transforms = transforms.Compose([
      transforms.ToTensor(),
  ])
  
  ##collect images from results folder
  image_ids=[]
  image_dir=results_dir+CGAN_model_name+"/"+CGAN_test_name+"/"
  for root,_,files in os.walk(image_dir):
    image_ids.extend(files)

  ##only keep the fake_B images, which are domain A images stylised in to domain B
  image_ids=[id for id in image_ids if "fake_B" in id]
  print("images found: ",len(image_ids))
      
  ##take correct number of images    
  if max_patches<len(image_ids):
    random.shuffle(image_ids)
    image_ids=image_ids[:max_patches]

  ##get model predictions
  bboxs=[]
  true_bboxs=[]
  for a in range(math.ceil(len(image_ids)/500)):
    _,test_loader,_ = create_data_loaders(image_ids[:10],image_ids[a*500:(a+1)*500],test_ids=image_ids,image_transform=image_transforms,
                                                            image_dir= image_dir,shuffle_valid=False,
                                                            train_transform=train_transforms,get_mask_name=mask_name_func,image_end_path=image_name_func)
    preds,masks,images=get_model_predictions(model,model_type,test_loader,device,max_batches=125,anchors=anchors)
    
    if model_type=="unet":
      bboxs.extend(add_bbox_to_map(preds,threshold=threshold,min_dim=min_detection_dim,return_images=False))
    elif model_type=="retinanet":
      bboxs.extend(preds)
      
    true_bboxs.extend(add_bbox_to_map(masks,threshold=threshold,return_images=False))
    
  return bboxs,true_bboxs



def get_nst_prediction(model,model_type,train_scanners,valid_scanners,image_transforms=None,train_transforms=None,anchors=None,
                       threshold=0.5,min_detection_dim=10,save_bboxs_name=None,nst_epochs=101,models_dir='/drive/MyDrive/MIDOG_Style_Transfer/models/',evalset='train',device='cuda',max_patches=1000):
  assert model_type in ["unet","retinanet"], "model_type must be unet or retinanet"
  
  vgg = models.vgg19(pretrained = True).features ##removes classifier
  for parameters in vgg.parameters():
    parameters.requires_grad_(False)
  vgg.to(device)

  if train_transforms==None:
    train_transforms = transforms.Compose([
    ])
  if image_transforms==None:
    image_transforms = transforms.Compose([
      transforms.ToTensor(),
    ])

  if evalset=='train':
    style_ids,image_ids,_ = preprocess_crops(max_patches,max_patches,10,train_scanners=train_scanners,valid_scanners=valid_scanners)
  elif evalset=='test':
    style_ids,_,image_ids = preprocess_crops(max_patches,10,max_patches,train_scanners=train_scanners,valid_scanners=valid_scanners)

  bboxs=[]
  true_bboxs=[]
  for a in range(math.ceil(len(image_ids)/500)):
    _,valid_loader = create_data_loaders_nst(style_ids[a*500:(a+1)*500],image_ids[a*500:(a+1)*500],vgg,device,image_transform=image_transforms,train_transform=train_transforms,nst_epochs=nst_epochs,shuffle_valid=False)
    preds,masks,images=get_model_predictions(model,model_type,valid_loader,device,max_batches=125,anchors=anchors)
    
    if model_type=="unet":
      bboxs.extend(add_bbox_to_map(preds,threshold=threshold,min_dim=min_detection_dim,return_images=False))
    elif model_type=="retinanet":
      bboxs.extend(preds)

    true_bboxs.extend(add_bbox_to_map(masks,threshold=threshold,return_images=False))
    
    if save_bboxs_name!=None:
      with open(models_dir+save_bboxs_name+'bbox.txt', 'w') as f:
        f.write(json.dumps(bboxs))
      with open(models_dir+save_bboxs_name+'true_bbox.txt', 'w') as f:
        f.write(json.dumps(true_bboxs))
    print("Rolling F1-score: ",F1_score_centered(bboxs,true_bboxs))

  return bboxs,true_bboxs



def bootstrap_f1_scores(prediction_bboxs,true_bboxs,bootstraps=10000,patches_per_bootstrap=-1):
  assert len(prediction_bboxs)==len(true_bboxs), "input length mismatch"

  if patches_per_bootstrap==-1:
    patches_per_bootstrap=len(prediction_bboxs)

  F1_scores=[]
  for j in tqdm(range(bootstraps)):
    idxs=np.random.choice(range(len(prediction_bboxs)),patches_per_bootstrap)
    F1_scores.append(F1_score_centered([prediction_bboxs[i] for i in idxs],[true_bboxs[i] for i in idxs])[0])
  return F1_scores
