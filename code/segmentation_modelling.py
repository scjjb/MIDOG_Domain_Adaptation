##Some of this code is adapted from https://github.com/amaarora/amaarora.github.io/blob/master/nbs/Training.ipynb

from pathlib import Path
from tqdm import tqdm_notebook as tqdm
import numpy as np
import torch
import os
from sklearn.neighbors import KDTree
from torchvision import transforms
import math
from fastai.vision import tensor
from .segmentation_processing import MIDOG_Images,get_mask_name,image_end_path,mask_end_path
from .data_funcs import get_bbox_df
from torch.utils.data import WeightedRandomSampler



def create_data_loaders(train_ids,valid_ids,image_transform=None,mask_transform=None,train_transform=None,test_ids=None,shuffle_valid=True,
                        normalise="ImageNet",image_dir=None,mask_dir=None,get_mask_name=get_mask_name,image_end_path=image_end_path,mask_end_path=mask_end_path,weight_multiplier=5):

  if normalise == "ImageNet":
    norm_mean=[0.485, 0.456, 0.406]
    norm_sd=[0.229, 0.224, 0.225]
  if normalise == "Hamamatsu XR":
    norm_mean=[197.53/255,143.54/255,202.30/255]
    norm_sd=[math.sqrt(690.74)/255,math.sqrt(1279.30)/255,math.sqrt(237.16)/255]
  if normalise == "Hamamatsu S360":
    norm_mean=[206.11/255,144.28/255,187.62/255]
    norm_sd=[math.sqrt(670.45)/255,math.sqrt(1522.41)/255,math.sqrt(601.91)/255]
  if normalise == "Aperio CS":
    norm_mean=[202.74/255,149.97/255,174.83/255]
    norm_sd=[math.sqrt(731.78)/255,math.sqrt(1480.70)/255,math.sqrt(855.76)/255]
  if normalise == "Leica GT450":
    norm_mean=[231.52/255,197.26/255,230.18/255]
    norm_sd=[math.sqrt(317.51)/255,math.sqrt(797.85)/255,math.sqrt(134.20)/255]

  if image_transform is None:
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(tensor(norm_mean),tensor(norm_sd))
    ])
    
  if mask_transform is None:
    mask_transform = transforms.Compose([
        transforms.ToTensor(),           
    ])


  dataset_train = MIDOG_Images(
      image_ids=train_ids, image_transform=image_transform, mask_transform=mask_transform,train_transform=train_transform,
      image_dir=image_dir,mask_dir=mask_dir,get_mask_name=get_mask_name,image_end_path=image_end_path,mask_end_path=mask_end_path,
  )

  boxes_df=get_bbox_df()
  weight=np.ones(len(train_ids))*0.01
  for index,row in boxes_df.iterrows():
    id=row['file_name']
    x=int(row['point'][0])
    x=math.floor(x/512)*512
    y=int(row['point'][1])
    y=math.floor(y/512)*512
    image_name=str(id[0:3])+"_x"+str(x)+"_y"+str(y)+".tiff"
    if image_name in train_ids:
      weight[train_ids.index(image_name)]=weight_multiplier*0.01
  sampler = WeightedRandomSampler(weight, len(weight))
  
  train_loader = torch.utils.data.DataLoader(
      dataset_train, batch_size=8, num_workers=4, sampler=sampler,
  )

  dataset_valid = MIDOG_Images(
      image_ids=valid_ids, image_transform=image_transform, mask_transform=mask_transform,
      image_dir=image_dir,mask_dir=mask_dir,get_mask_name=get_mask_name,image_end_path=image_end_path,mask_end_path=mask_end_path,
  )

  valid_loader = torch.utils.data.DataLoader(
      dataset_valid, batch_size=8, shuffle=shuffle_valid, num_workers=4,
  )

  if test_ids is not None:
    dataset_test = MIDOG_Images(
        image_ids=test_ids, image_transform=image_transform, mask_transform=mask_transform,
        image_dir=image_dir,mask_dir=mask_dir,get_mask_name=get_mask_name,image_end_path=image_end_path,mask_end_path=mask_end_path,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=8, shuffle=shuffle_valid, num_workers=4,
    )

    return train_loader, valid_loader, test_loader
  
  return train_loader, valid_loader



class EarlyStopping:  
    def __init__(self, patience=7, mode="max", delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        model_path = Path(model_path)
        parent = model_path.parent
        os.makedirs(parent, exist_ok=True)
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Model saved at at {}!".format(
                    self.val_score, epoch_score, model_path
                )
            )
            torch.save(model, model_path)
        self.val_score = epoch_score
        
        
        
class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
        
def train_one_epoch(train_loader, model, optimizer, loss_fn, batches_per_epoch=125, accumulation_steps=1, device='cuda'):
    losses = AverageMeter()
    model = model.to(device)
    model.train()
    if accumulation_steps > 1: 
        optimizer.zero_grad()
    tk0 = tqdm(train_loader, total=min(batches_per_epoch,len(train_loader)))
    for b_idx, data in enumerate(tk0):
        if b_idx>=batches_per_epoch:
          break

        for key, value in data.items():
            if (str(type(value))=="<class 'torch.Tensor'>"):
              data[key] = value.to(device)
            
        if accumulation_steps == 1 and b_idx == 0:
            optimizer.zero_grad()
        out  = model(data['image'])
        loss = loss_fn(out, data['mask'])
        with torch.set_grad_enabled(True):
            loss.backward()
            if (b_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        losses.update(loss.item(), train_loader.batch_size)
        tk0.set_postfix(loss=losses.avg, learning_rate=optimizer.param_groups[0]['lr'])
    return losses.avg





def metric(probability, truth, threshold=0.5, reduction='none'):
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice
  
  

def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


  
def evaluate(valid_loader, model, device='cuda', metric=dice_loss):
    losses = AverageMeter()
    model = model.to(device)
    model.eval()
    tk0 = tqdm(valid_loader, total=len(valid_loader))
    for b_idx, data in enumerate(tk0):
        with torch.no_grad():
            for key, value in data.items():
                if (str(type(value))=="<class 'torch.Tensor'>"):
                    data[key] = value.to(device)
            out   = model(data['image'])
            out   = torch.sigmoid(out)
            dice  = metric(out, data['mask']).cpu()
            losses.update(dice.mean().item(), valid_loader.batch_size)
            tk0.set_postfix(dice_score=losses.avg)
    return losses.avg



def F1_score_centered(predictions,ground_truths, radius = 30):
  F1 = 0
  TP = 0
  FP = 0
  FN = 0

  for i in range(len(predictions)):
    x_pred_collector=[]
    y_pred_collector=[]

    for j in range(len(ground_truths[i])):
      x_pred_collector.append(ground_truths[i][j][0])
      y_pred_collector.append(ground_truths[i][j][1])

    for j in range(len(predictions[i])):
      x_pred_collector.append(predictions[i][j][0])
      y_pred_collector.append(predictions[i][j][1])

    isDet = np.zeros(len(predictions[i])+len(ground_truths[i]))
    isDet[0:len(ground_truths[i])]=1

    X=np.dstack((x_pred_collector, y_pred_collector))[0]
    annotationWasDetected = {}
    DetectionMatchesAnnotation = {}
    if X.shape[0]>0:
      tree = KDTree(X)
      ind = tree.query_radius(X, r=radius)

      annotationWasDetected = {x: 0 for x in np.where(isDet==0)[0]}
      DetectionMatchesAnnotation = {x: 0 for x in np.where(isDet==1)[0]}

      for m in ind:
        if len(m)==0:
            continue

        if np.any(isDet[m]) and np.any(isDet[m]==0):
            # at least 1 detection and 1 non-detection --> count all as hits
            for j in range(len(m)):
                if not isDet[m][j]: # is annotation, that was detected
                    if m[j] not in annotationWasDetected:
                        print('Missing key ',j, 'in annotationWasDetected')
                        raise ValueError('Ijks')
                    annotationWasDetected[m[j]] = 1
                else:
                    if m[j] not in DetectionMatchesAnnotation:
                        print('Missing key ',j, 'in DetectionMatchesAnnotation')
                        raise ValueError('Ijks')
                    DetectionMatchesAnnotation[m[j]] = 1

      TP = TP + np.sum([annotationWasDetected[x]==1 for x in annotationWasDetected.keys()])
      FP = FP + np.sum([annotationWasDetected[x]==0 for x in annotationWasDetected.keys()])
      FN = FN + np.sum([DetectionMatchesAnnotation[x]==0 for x in DetectionMatchesAnnotation.keys()])

  try:
    F1 = 2*TP/(2*TP + FP + FN)
  except:
    F1 = 0

  return [round(F1,3), int(TP), int(FP), int(FN)]
