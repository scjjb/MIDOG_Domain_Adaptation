import numpy as np
import math
from tqdm import tqdm_notebook as tqdm
from PIL import Image 
import os 
import glob

def create_mask(image_no,df,categories=["hard negative","mitotic figure"]):
  box_df=df[df['image_id']==image_no].reset_index(drop=True)
  if box_df.shape[0] > 0:
    bin_mask=np.zeros([box_df['height'][0],box_df['width'][0]], dtype=int)
    for index, row in box_df.iterrows():
      if row['cat'] in categories:
        ##had to add these min/maxs to the below rows as some boxes go past the edges of the image
        for i in range(max(math.floor(row['box'][0]),0),min(box_df['width'][0],math.floor(row['box'][2]))):
          for j in range(max(math.floor(row['box'][1]),0),min(box_df['height'][0],math.floor(row['box'][3]))): 
            bin_mask[j,i]=int(1)
  return bin_mask



def mask_segmentor(image_ids,df,segment_height=600,segment_width=600,
                   input_folder="/drive/MyDrive/MIDOG_Challenge/images/",
                   image_folder="/drive/MyDrive/MIDOG_Challenge_JJB/image_crops/",
                   mask_folder="/drive/MyDrive/MIDOG_Challenge_JJB/mask_crops/",
                   categories=["hard negative","mitotic figure"]):
  prepare_folders(image_folder,mask_folder)
  
  for id in tqdm(image_ids):
    image = Image.open(input_folder+id).convert('RGB')
    bin_mask = create_mask(int(id[0:3]),df,categories=categories)
    mask_img = Image.fromarray((bin_mask * 255).astype(np.uint8))
    imshape=image.size
    
    for i in range(0,math.floor(imshape[0]/segment_width)):
      min_x=i*segment_width
      max_x=(i+1)*segment_width

      for j in range(0,math.floor(imshape[1]/segment_height)):
        min_y=j*segment_height
        max_y=(j+1)*segment_height
        cropped_im = image.crop((min_x, min_y, max_x, max_y))#left upper right lower
        cropped_mask = mask_img.crop((min_x, min_y, max_x, max_y))
        cropped_im.save(image_folder+str(id[0:3])+"/"+str(id[0:3])+"_x"+str(min_x)+"_y"+str(min_y)+".tiff")
        cropped_mask.save(mask_folder+str(id[0:3])+"/"+str(id[0:3])+"mask_x"+str(min_x)+"_y"+str(min_y)+".tiff")
        
        
        
def prepare_folders(image_folder="/drive/MyDrive/MIDOG_Challenge_JJB/image_crops/",
                     mask_folder="/drive/MyDrive/MIDOG_Challenge_JJB/mask_crops/"):
  if not os.path.isdir(image_folder):
    os.makedirs(image_folder)
  if not os.path.isdir(mask_folder):
    os.makedirs(mask_folder)

  scanner_ids=[str(i+1).zfill(3) for i in range(150)]
  for id in scanner_ids:
    if os.path.isdir(image_folder+id):
      files = glob.glob(image_folder+id+"/*")
      for file in files:
        os.remove(file)
    else:
      os.makedirs(image_folder+id)
    if os.path.isdir(mask_folder+id):
      files = glob.glob(mask_folder+id+"/*")
      for file in files:
        os.remove(file)
    else:
      os.makedirs(mask_folder+id)
