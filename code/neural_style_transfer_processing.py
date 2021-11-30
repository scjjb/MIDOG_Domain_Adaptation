from PIL import Image 
from fastai.vision import tensor
from torchvision import transforms
import numpy as np



def preprocess(img_path, max_size = 1500,input_is_path=True): 
  if input_is_path:
    image = Image.open(img_path).convert('RGB')
  else:
    image=transforms.ToPILImage()(img_path.data).convert("RGB")
    
  if max(image.size) > max_size:
    size = max_size
  else:
    size = max(image.size)

  img_transforms = transforms.Compose([
                              transforms.Resize(size),
                              transforms.ToTensor(),
                              ##normalise using VGG pretraining normalisation 
                              transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224, 0.225]),
  ])
  image = img_transforms(image)
  image = image.unsqueeze(0) ## (3,224,224) -> (1,3,224,224) etc

  return image



def deprocess(tensor):
  ##function to undo preprocessing step
  image = tensor.to('cpu').clone() ##make a copy for later
  image = image.numpy() 
  image = image.squeeze(0)
  image = image.transpose(1,2,0) ##required as ToTensor changes (224,224,3) to (3,224,224)
  image = image * np.array([0.229,0.224, 0.225]) +np.array([0.485,0.456,0.406])
  image = image.clip(0,1)

  return(image)
