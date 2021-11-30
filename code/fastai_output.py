##predictions functions from https://www.kaggle.com/ianmoone0617/fastai-v1-global-wheat-detection?scriptVersionId=41013238
##which is strongly tied to Marzahls code
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from object_detection_fastai.helper.object_detection_helper import draw_rect, rescale_boxes, process_output, nms
from MIDOG.Code.neural_style_transfer_modelling import style_transfer
from fastai.vision import torch, image2np, LearnerCallback,Image
from fastai.torch_core import to_np
import random
from fastai.vision import tensor
from tqdm import tqdm_notebook as tqdm



def show_output(item,bboxs_tot,preds_tot,scores_tot,ground_truth,learn):
    fig,ax = plt.subplots(figsize=(10,10))
    ax.imshow(image2np(item.data))
    plt.axis('off')
    area_max = 512**2/5 
    classes=learn.data.train_ds.classes[1:]
    if len(scores_tot)>0:
      for bbox, c, scr in zip(bboxs_tot[0], preds_tot[0], scores_tot[0].numpy()):
          txt = str(c.item()) if classes is None else classes[c.item()]
          #if bbox[2]*bbox[3] <= area_max:
          draw_rect(ax, [bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{txt} {scr:.2f}',text_size=12,color='red')
    if ground_truth.data[1][0]>0:
      t_sz = torch.Tensor([item.size])[None].cpu()
      bboxs = ground_truth.data[0].cpu()
      bboxs[:, 2:] = bboxs[:, 2:] - bboxs[:, :2]
      bboxs = to_np(rescale_boxes(bboxs, t_sz))
      for i in range(len(bboxs)):
        bbox = bboxs[i]
        c=ground_truth.data[1][i]  
        txt = str(c.item()) if classes is None else classes[c.item()-1]
        draw_rect(ax, [bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{txt}')



def process_preds_show(item,clas,bboxs,ground_truth,learn,anchors,show_img,cnt,i,detect_threshold=0.2,nms_threshold=0.2):
    pred_string = []
    scores_tot = []
    bboxs_tot = []
    preds_tot = []
    pred_list = []
    #show_img = True if i<cnt else False
    for clas_pred, bbox_pred in list(zip(clas, bboxs)):
        bbox_pred, scores, preds = process_output(clas_pred, bbox_pred, anchors, detect_threshold)
        if bbox_pred is not None:
            to_keep = nms(bbox_pred, scores, nms_threshold)
            bbox_pred, preds, scores = bbox_pred[to_keep].cpu(), preds[to_keep].cpu(), scores[to_keep].cpu()
        t_sz = torch.Tensor([item.size])[None].cpu()
        if bbox_pred is not None:
            bbox_pred = to_np(rescale_boxes(bbox_pred, t_sz))
            # change from center to top left
            bbox_pred[:, :2] = bbox_pred[:, :2] - bbox_pred[:, 2:] / 2
            bboxs_tot.append(bbox_pred)
            preds_tot.append(preds)
            scores_tot.append(scores)
            
    if show_img:
        show_output(item,bboxs_tot,preds_tot,scores_tot,ground_truth,learn)
    area_max = (1024**2)/5
    if len(scores_tot)>0:
      for s,pred,bbx in zip(scores_tot[0].numpy(),preds_tot[0],bboxs_tot[0]):
          bbx = [int(round(x)) for x in bbx*2]
          if bbx[2]*bbx[3] <= area_max :
              pred_list.append([i,s,pred.item()+1,bbx[1],bbx[0],bbx[3],bbx[2]])
    return pred_list



def get_prediction(learn,anchors,show_img=True,cnt=10,epochs=101,styles=None,model=None,device=None,detect_threshold=0.2,nms_threshold=0.2): 
    #################
    ###
    ### Outputs predictions, ground truths with format:
    ### Certainty score, class, bbox coords
    ### Where class = 1 for hard negative, 2 for mitotic figure
    ### bbox coords = [x,y,width,height] with x,y coordinate of top left
    ###
    #################
    # Set show img True to see img or else false for bboxs only, cnt for number of images to show
    ground_truth_boxes = []
    preds_boxes = []
        
    iters=min(cnt,len(learn.data.valid_ds))
    for i in tqdm(range(iters),total=iters):
        item = learn.data.valid_ds[i]  #Pick one image and its bboxs
        batch = learn.data.one_item(item[0])
        ground_truth = item[1]
        clas,bboxs,xtr = learn.pred_batch(batch=batch)
        
        if styles is not None:
            style_image=random.choice(styles)
            styled_image=style_transfer(item[0],style_image,model,device,epochs=epochs,print_out=False,show_every=epochs-1,show_images=False,input_is_path=False)[-1]
            pred_list = process_preds_show(Image(tensor(styled_image.transpose(2,0,1))),clas,bboxs,ground_truth,learn,anchors,show_img,cnt,i,detect_threshold,nms_threshold) 
            
        else:
            pred_list = process_preds_show(item[0],clas,bboxs,ground_truth,learn,anchors,show_img,cnt,i,detect_threshold,nms_threshold)
        
        
        for k in range(len(pred_list)):
          preds_boxes.append(pred_list[k])

        if ground_truth.data[1][0]>0:
          t_sz = torch.Tensor([item[0].size])[None].cpu()
          bboxs_true = ground_truth.data[0].cpu()
          bboxs_true[:, 2:] = bboxs_true[:, 2:] - bboxs_true[:, :2]
          bboxs_true = to_np(rescale_boxes(bboxs_true, t_sz))
          for j in range(len(ground_truth.data[1])):
            coords = [int(round(x)) for x in bboxs_true[j]*2]
            ground_truth_boxes.append([i,1,int(ground_truth.data[1][j]),coords[1],coords[0],coords[3],coords[2]])

    pred_df = pd.DataFrame(preds_boxes,columns=["image","score","class","x","y","width","height"])
    true_df = pd.DataFrame(ground_truth_boxes,columns=["image","score","class","x","y","width","height"])

    return pred_df, true_df


def F1_score(predictions,ground_truths, radius = 30):
    highest_image_id=max(max(ground_truths['image'],default=0),max(predictions['image'],default=0))
    F1_mitot = 0
    TP_mitot = 0
    FP_mitot = 0
    FN_mitot = 0
    F1_hneg = 0
    TP_hneg = 0
    FP_hneg = 0
    FN_hneg = 0
    
    for i in range(highest_image_id+1):
      preds = predictions[predictions['image']==i].reset_index(drop=True)
      truths = ground_truths[ground_truths['image']==i].reset_index(drop=True)
        
      centered_preds = []
      centered_truths = []

      for class_ind in range(2):

        ## Select boxes from the same image with the same class
        preds = predictions[predictions['image']==i].reset_index(drop=True)
        preds = preds[preds['class']==(class_ind+1)].reset_index(drop=True)
        truths = ground_truths[ground_truths['image']==i].reset_index(drop=True)
        truths = truths[truths['class']==(class_ind+1)].reset_index(drop=True)

        x_center=[]
        y_center=[]

        for j in range(truths.shape[0]):
          x_center.append(truths['x'][j] + (truths['width'][j]/2))
          y_center.append(truths['y'][j] + (truths['height'][j]/2))
        
        for j in range(preds.shape[0]):
          x_center.append(preds['x'][j] + (preds['width'][j]/2))
          y_center.append(preds['y'][j] + (preds['height'][j]/2))

        isDet = np.zeros(preds.shape[0]+truths.shape[0])
        isDet[0:truths.shape[0]]=1

        X=np.dstack((x_center, y_center))[0]

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

          if (class_ind+1) == 1:
            TP_hneg = TP_hneg + np.sum([annotationWasDetected[x]==1 for x in annotationWasDetected.keys()])
            FP_hneg = FP_hneg + np.sum([annotationWasDetected[x]==0 for x in annotationWasDetected.keys()])
            FN_hneg = FN_hneg + np.sum([DetectionMatchesAnnotation[x]==0 for x in DetectionMatchesAnnotation.keys()])

          if (class_ind+1) == 2:
            TP_mitot = TP_mitot + np.sum([annotationWasDetected[x]==1 for x in annotationWasDetected.keys()])
            FP_mitot = FP_mitot + np.sum([annotationWasDetected[x]==0 for x in annotationWasDetected.keys()])
            FN_mitot = FN_mitot + np.sum([DetectionMatchesAnnotation[x]==0 for x in DetectionMatchesAnnotation.keys()])
    try:
      F1_mitot = 2*TP_mitot/(2*TP_mitot + FP_mitot + FN_mitot)
    except:
      F1_mitot = 0
    try:
      F1_hneg = 2*TP_hneg/(2*TP_hneg + FP_hneg + FN_hneg)
    except:
      F1_hneg = 0

        ## Need to check that single mitotic figure isnt being used for multiple detections
    return [round(F1_mitot,3), int(TP_mitot), int(FP_mitot), int(FN_mitot), round(F1_hneg,3), int(TP_hneg), int(FP_hneg), int(FN_hneg)]



class F1Metric(LearnerCallback):
  def on_train_begin(self, **kwargs):
    self.F1_score_df = pd.DataFrame(columns = ['F1_mitot','TP_mitot','FP_mitot','FN_mitot','F1_hneg','TP_hneg','FP_hneg','FN_hneg'])

  def on_epoch_end(self, **kwargs):
    pred_df, true_df = get_prediction(self.learn,self.learn.anchors,show_img=False,cnt=100) 
    self.F1_score_df.loc[self.F1_score_df.shape[0]] = F1_score(pred_df,true_df)

  def on_train_end(self,**kwargs):
    print(self.F1_score_df)
