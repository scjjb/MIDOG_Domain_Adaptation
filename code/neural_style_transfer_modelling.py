from PIL import Image 
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from fastai.vision import tensor
from .neural_style_transfer_processing import preprocess, deprocess
from .segmentation_processing import MIDOG_Images,get_mask_name,image_end_path,mask_end_path
from torch.utils.data import Dataset
import random



def get_features(image,model):
  layers = {
      '0' : 'conv1_1',
      '5' : 'conv2_1',
      '10' : 'conv3_1',
      '19' : 'conv4_1',
      '21' : 'conv4_2', #content_feature
      '28' : 'conv5_1'
  }

  x=image
  Features = {}
  for name,layer in model._modules.items():
    x = layer(x)
    if name in layers:
      Features[layers[name]] = x
  return Features



def gram_matrix(tensor):
  b,c,h,w = tensor.size()
  tensor = tensor.view(c,h*w)
  gram = torch.mm(tensor,tensor.t())
  return gram



def content_loss(target_conv4_2,content_conv4_2):
  loss = torch.mean((target_conv4_2-content_conv4_2)**2)
  return loss



def style_loss(style_weights,target_features,style_grams):
  loss = 0
  for layer in style_weights:
    target_f = target_features[layer]
    target_gram = gram_matrix(target_f)
    style_gram = style_grams[layer]
    b,c,h,w = target_f.shape
    layer_loss = style_weights[layer] * torch.mean((target_gram-style_gram)**2)
    loss += layer_loss/(c*h*w)
  return loss



def total_loss(c_loss,s_loss,alpha,beta):
  loss = alpha * c_loss + beta * s_loss
  return loss



def style_transfer(content_path,style_path,model,device,epochs=101,print_out=True,show_every=20,alpha=1,beta=1e5,show_images=True, max_size = 1500,input_is_path=True):

  content = preprocess(content_path,max_size,input_is_path=input_is_path).to(device)
  style= preprocess(style_path,max_size).to(device)

  target = content.clone().requires_grad_(True).to(device)
  optimizer = torch.optim.Adam([target],lr=0.003)

  content_f = get_features(content,model)
  style_f = get_features(style,model)
  style_grams={layer : gram_matrix(style_f[layer]) for layer in style_f}

  style_weights = {
    'conv1_1' : 1.0,
    'conv2_1' : 0.75,
    'conv3_1' : 0.2,
    'conv4_1' : 0.2,
    'conv5_1' : 0.2,
  }

  results = []
  for i in range(epochs):
    target_f = get_features(target,model)
    c_loss = content_loss(target_f['conv4_2'],content_f['conv4_2']).requires_grad_(True)
    s_loss = style_loss(style_weights,target_f,style_grams).requires_grad_(True)
    t_loss = total_loss(c_loss,s_loss,alpha,beta).requires_grad_(True)

    optimizer.zero_grad()
    t_loss.backward()
    optimizer.step()

    if i % show_every == 0:
      results.append(deprocess(target.detach()))
      if print_out==True:
        print("Total Loss at Epoch {} : {}".format(i,t_loss))

  if show_images==True:
    plt.figure(figsize = (30,20))
    for i in range(len(results)):
      plt.subplot(len(results),1,i+1)
      plt.imshow(results[i])
    plt.show()
  return results



##Following code is adjusted from equivalent non-nst versions
class MIDOG_Images_NST(Dataset):
    
    def __init__(self, image_ids,train_ids,model,device,nst_epochs=101,image_transform=None,mask_transform=None,train_transform=None,keep_id=True,
                 image_dir=None,mask_dir=None,get_mask_name=get_mask_name,image_end_path=image_end_path,mask_end_path=mask_end_path,beta=1e5):

        self.image_ids = image_ids
        self.train_ids = train_ids
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.train_transform = train_transform
        self.keep_id = keep_id
        self.image = []
        self.mask = []
        self.get_mask_name=get_mask_name
        self.image_end_path=image_end_path
        self.mask_end_path=mask_end_path
        self.model=model
        self.device=device
        self.nst_epochs=nst_epochs
        self.beta=beta
        
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
        style_name = random.choice(self.train_ids)
        mask_name=self.get_mask_name(image_name)
        image_path = self.image_dir + self.image_end_path(image_name)
        style_path = self.image_dir + self.image_end_path(style_name)
        mask_path = self.mask_dir + self.mask_end_path(mask_name)
        mask = Image.open(open(mask_path, 'rb'))

        image = style_transfer(image_path,style_path,self.model,self.device,epochs=self.nst_epochs,
                               print_out=False,show_every=self.nst_epochs-1,show_images=False,max_size=512,beta=self.beta)[-1]

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
        
        image = image.to(torch.float32)
        if self.keep_id == True:
          return {'image': image, 'mask' : mask, 'id': image_name}
        if self.keep_id == False:
          return {'image': image, 'mask' : mask}

    def __len__(self):
        return len(self.image_ids)

   
  
def create_data_loaders_nst(train_ids,valid_ids,model,device,nst_epochs=101,image_transform=None,mask_transform=None,train_transform=None,test_ids=None,shuffle_valid=True,
                        normalise="ImageNet",image_dir=None,mask_dir=None,get_mask_name=get_mask_name,image_end_path=image_end_path,mask_end_path=mask_end_path,beta=1e5):

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

  train_loader = torch.utils.data.DataLoader(
      dataset_train, batch_size=8, shuffle=True, num_workers=4,
  )

  dataset_valid = MIDOG_Images_NST(
      image_ids=valid_ids,train_ids=train_ids,nst_epochs=nst_epochs,model=model,device=device,image_transform=image_transform, mask_transform=mask_transform,
      image_dir=image_dir,mask_dir=mask_dir,get_mask_name=get_mask_name,image_end_path=image_end_path,mask_end_path=mask_end_path,beta=beta
  )

  valid_loader = torch.utils.data.DataLoader(
      dataset_valid, batch_size=8, shuffle=shuffle_valid, num_workers=0,
  )

  if test_ids is not None:
    dataset_test = MIDOG_Images_NST(
        image_ids=test_ids, train_ids=train_ids,nst_epochs=nst_epochs,model=model,device=device, image_transform=image_transform, mask_transform=mask_transform,
        image_dir=image_dir,mask_dir=mask_dir,get_mask_name=get_mask_name,image_end_path=image_end_path,mask_end_path=mask_end_path,beta=beta
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=8, shuffle=shuffle_valid, num_workers=0,
    )

    return train_loader, valid_loader, test_loader
  
  return train_loader, valid_loader



class NST(object):
    def __init__(self, style_images,model,device,beta=1e5):
        assert isinstance(style_images, (list))
        self.style_images = style_images
        self.model = model
        self.device = device
        self.beta = beta
    
    def __call__(self, sample):
        images, other = sample[0], sample[1]

        for image in images:
          style_image=random.choice(self.style_images)
          styled_image=style_transfer(image,style_image,self.model,self.device,print_out=False,show_every=100,show_images=False,input_is_path=False,beta=self.beta)[-1]
          styled_image=styled_image.transpose(2,0,1)           
          styled_image=torch.unsqueeze(tensor(styled_image),0)
          
          try:
            imgs = torch.cat((imgs, styled_image), 0)
          except:
            imgs=styled_image
          
        return tuple([imgs, other])
