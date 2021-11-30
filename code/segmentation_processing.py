import numpy as np
from os import listdir
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from PIL import Image
import torch
import torchvision
import cv2
import random
from .fastai_sampling import activ_to_bbox, nms_patch, rescale_box, cthw2tlbr, tlbr2cthw


def preprocess_crops(no_of_train=5000,no_of_valid=5000,no_of_test=5000,randomise_ids=True,
                     train_scanners=["Hamamatsu XR"],valid_scanners=["Hamamatsu S360"],
                     image_dir="/drive/MyDrive/MIDOG_Challenge_JJB/image_crops/",
                     mask_dir="/drive/MyDrive/MIDOG_Challenge_JJB/mask_crops/"):
  scanner_ids=[str(i+1).zfill(3) for i in range(150)]
  all_images=[]
  all_masks=[]
  for id in scanner_ids:
    all_images.extend(listdir(image_dir+id+"/"))

  train_scanner_ids=[]
  for scanner in train_scanners:  
    if scanner == "Hamamatsu XR":
      adjustment=1
    if scanner == "Hamamatsu S360":
      adjustment=51
    if scanner == "Aperio CS":
      adjustment=101
    for i in range(40):
      train_scanner_ids.append(str(i+adjustment).zfill(3))
  
  
  valid_scanner_ids=[]
  test_scanner_ids=[]
  for scanner in valid_scanners:
    if scanner == "Hamamatsu XR":
      adjustment_val=1
    if scanner == "Hamamatsu S360":
      adjustment_val=51
    if scanner == "Aperio CS":
      adjustment_val=101
    for i in range(40):
      valid_scanner_ids.append(str(i+adjustment_val).zfill(3))
    for i in range(10):
      test_scanner_ids.append(str(i+adjustment_val+40).zfill(3))
    ##test images (final 10 of each scanner) are not to be used for tuning models

  train_ids=[]
  valid_ids=[]
  test_ids=[]
  for image in all_images:
    if image[:3] in train_scanner_ids:
      train_ids.append(image)
    if image[:3] in valid_scanner_ids:
      valid_ids.append(image)  
    if image[:3] in test_scanner_ids:
      test_ids.append(image)  

  if no_of_train<len(train_ids):
    if randomise_ids==True:
      train_ids=random.sample(train_ids,no_of_train)
    else:
      train_ids=train_ids[:no_of_train] 
  else:
    print("only {} training images available".format(len(train_ids)))
    
  if no_of_valid<len(valid_ids):
    if randomise_ids==True:
      valid_ids=random.sample(valid_ids,no_of_valid)
    else:
      valid_ids=valid_ids[:no_of_valid]
  else:
    print("only {} validation images available".format(len(valid_ids)))
    
  if no_of_test<len(test_ids):
    if randomise_ids==True:
      test_ids=random.sample(test_ids,no_of_test)
    else:
      test_ids=test_ids[:no_of_test]
  else:
    print("only {} test images available".format(len(test_ids)))

  return train_ids,valid_ids,test_ids



def show_images_in_grid(images,max_images=16):
    x=torchvision.utils.make_grid(images[:max_images])
    xa = np.transpose(x.numpy(),(1,2,0))
    plt.imshow(xa)
    plt.show()
    
      
      
def visualize(images,positives_only=False):
    """Plot images in one row."""
    images = {k:v.numpy() for k,v in images.items() if isinstance(v, torch.Tensor)} #convert tensor to numpy 
    n = len(images)
    image, mask = images['image'], images['mask']
    if positives_only:
      if mask.max()>0:
        plt.figure(figsize=(16, 8))
        plt.imshow(image.transpose(1,2,0), vmin=0, vmax=1)
        plt.imshow(mask.squeeze(0), alpha=0.25)
        plt.show()
    else:
      plt.figure(figsize=(16, 8))
      plt.imshow(image.transpose(1,2,0), vmin=0, vmax=1)
      if mask.max()>0:
        plt.imshow(mask.squeeze(0), alpha=0.25)
      plt.show()      
      

      
def get_mask_name(image_name):
  return image_name[0:3]+"mask"+image_name[3:]      

def image_end_path(image_name):
  return str(image_name)[:3] + "/" + str(image_name)
  
def mask_end_path(mask_name):
  return str(mask_name)[:3] + "/" + str(mask_name)
  
      
class MIDOG_Images(Dataset):
    
    def __init__(self, image_ids, image_transform=None,mask_transform=None,train_transform=None,keep_id=True,
                 image_dir=None,mask_dir=None,get_mask_name=get_mask_name,image_end_path=image_end_path,mask_end_path=mask_end_path):
        """ Set the path for images, captions and vocabulary wrapper.
        
        Args:
            image_ids (str list): list of image ids
            transform: image transformer
        """
        self.image_ids = image_ids
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.train_transform = train_transform
        self.keep_id = keep_id
        self.image = []
        self.mask = []
        self.get_mask_name=get_mask_name
        self.image_end_path=image_end_path
        self.mask_end_path=mask_end_path
        
        if image_dir == None:
          self.image_dir = "/drive/MyDrive/MIDOG_Challenge_JJB/image_crops/"
        else:
          self.image_dir = image_dir
          
        if mask_dir == None:
          self.mask_dir = "/drive/MyDrive/MIDOG_Challenge_JJB/mask_crops/"
        else:
          self.mask_dir = mask_dir
        
        
    def __getitem__(self, index):
        """ Returns image. """

        image_name = self.image_ids[index]
        mask_name=self.get_mask_name(image_name)
        image_path = self.image_dir + self.image_end_path(image_name)
        mask_path = self.mask_dir + self.mask_end_path(mask_name)
        image = Image.open(open(image_path, 'rb'))
        mask = Image.open(open(mask_path, 'rb'))

        if self.image_transform is not None:
          image = self.image_transform(image)
        if self.mask_transform is not None:
          mask = self.mask_transform(mask)
        if self.train_transform is not None:
          stacked = torch.cat([image, mask], dim=0)  # shape=(2xHxW)
          stacked = self.train_transform(stacked)
          # Split them back up again
          image1,image2,image3,mask=torch.chunk(stacked, chunks=4, dim=0)
          image=torch.cat([image1,image2,image3],dim=0)

        if self.keep_id == True:
          return {'image': image, 'mask' : mask, 'id': image_name}
        if self.keep_id == False:
          return {'image': image, 'mask' : mask}

    def __len__(self):
        return len(self.image_ids)
      
      
def get_model_predictions(model,model_type,loader,device,max_batches=300,threshold=0.5,anchors=None,nms_threshold=0.5):
    assert model_type in ["unet","retinanet"], "model_type must be unet or retinanet"
    
    pred_collector =[]
    mask_collector=[]
    image_collector=[]

    iteration_len=min(max_batches,len(loader))
    
    
    for i, data in tqdm(enumerate(loader, 0),total=iteration_len):
      inputs, labels = data['image'],data['mask']
      inputs=inputs.to(device)
      labels=labels.to(device)

      with torch.no_grad():
        outputs = model(inputs)
        
      if model_type=="unet":
        prob_map = torch.sigmoid(outputs)
        pred_collector.extend(prob_map.data.squeeze().cpu().detach().clone().numpy())
      
      else:
        for j in range(inputs.shape[0]):
          bbox_pred=activ_to_bbox(outputs[1][j],anchors.to(device))
          clas_pred = torch.sigmoid(outputs[0][j])
          clas_pred_orig = clas_pred.clone()
          detect_mask = clas_pred.max(1)[0] > threshold
          bbox_pred, clas_pred = bbox_pred[detect_mask], clas_pred[detect_mask]
          if len(bbox_pred) > 0:
            
            # Perform nms per patch, as retinanet is prone to multiple predictions in the same location
            scores, preds = clas_pred.max(1)
            bbox_pred = tlbr2cthw(torch.clamp(cthw2tlbr(bbox_pred), min=-1, max=1))
            to_keep = nms_patch(bbox_pred, scores, nms_threshold)
            t_sz = torch.Tensor([[512, 512]]).float()
            bbox_pred = rescale_box(bbox_pred[to_keep].cpu(), t_sz)

            bboxs=[]
            for bbox in bbox_pred:
              bboxs.extend([[float(bbox[1]+(bbox[3]/2)),float(bbox[0]+(bbox[2]/2))]])
            pred_collector.extend([bboxs])

          else:
            pred_collector.extend([[]])
        
      mask_collector.extend(labels.data.squeeze().cpu().detach().clone().numpy())
      image_collector.extend(inputs.data.squeeze().cpu().detach().clone().numpy())
      if i >=max_batches:
        break
    return(pred_collector,mask_collector,image_collector)
      
    
    
def add_bbox_to_map(maps,threshold=0.3,min_dim=10,return_images=True):
  output_collector=[]
  
  for i in range(len(maps)):
    image = ((maps[i] >= threshold) * maps[i]*255).astype('uint8')

    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    image_midpoints=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)

        if w>min_dim and h>min_dim:
          midpoints=[x+(w/2),y+(h/2)]

          if return_images==True:
            image=cv2.rectangle(image, (int(midpoints[0]-25), int(midpoints[1]-25)), (int(midpoints[0]+25), int(midpoints[1]+25)), (255,255,255), 5)
          else:
            image_midpoints.append(midpoints)
            
    if return_images==True:
      output_collector.append(image)
    else:
      output_collector.append(image_midpoints)
  return output_collector
