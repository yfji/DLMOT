import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
from torch.autograd.variable import Variable as Variable

import numpy as np
import cv2

from model.mot_forward import MotFRCNN
from model.mot_forward import nms_cuda
import rpn.generate_anchors as G
from rpn.util import bbox_transform_inv
from rpn.generate_proposals import select_proposals_cosine
from core.config import cfg

import logging
logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(levelname)s - %(message)s')

CLASSES=['__background', 'car', 'van', 'bus', 'truck']
COLORS=[None, (0,0,255), (0,255,0), (255,0,0), (0,255,255)]

def get_detect_output(output_dict, batch_size=-1):
    frcnn_probs=output_dict['frcnn_probs'].cpu().data.numpy()
    frcnn_bbox=output_dict['frcnn_bbox'].cpu().data.numpy()

    proposals= output_dict['proposals']
    num_proposals_per_image=output_dict['num_proposals']
    
#    print(num_proposals_per_image)
#    print(frcnn_probs.shape)
    
    outputs=[]
    B=len(num_proposals_per_image)
    if batch_size>0:
        B=batch_size
    start=0
    for b in range(B):
        ret={}
        cls_bboxes=[[]]
        frcnn_probs_this_sample=frcnn_probs[start:start+num_proposals_per_image[b]]
        frcnn_bbox_this_sample=frcnn_bbox[start:start+num_proposals_per_image[b]]
        proposals_this_sample=proposals[start:start+num_proposals_per_image[b]]

        classes=np.argmax(frcnn_probs_this_sample, axis=1)
        max_probs=np.max(frcnn_probs_this_sample, axis=1)
#        print(classes)
        for i in range(1, cfg.NUM_CLASSES):
            cls_inds=np.where(classes==i)[0]
            if cls_inds.size==0:
                cls_bboxes.append(np.array([]))
            else:
                cls_proposals=proposals_this_sample[cls_inds]
                cls_frcnn_bbox=frcnn_bbox_this_sample[cls_inds,4*i:4*i+4]
                
                if cfg.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    cls_frcnn_bbox = cls_frcnn_bbox*cfg.BBOX_STD_DEV
                    
                cls_bbox_pred=bbox_transform_inv(cls_proposals, cls_frcnn_bbox)
                cls_probs=max_probs[cls_inds]
                
#                print(frcnn_probs_this_sample[cls_inds])
#         
                cls_bbox_pred=np.hstack((cls_bbox_pred, cls_probs.reshape(-1,1)))
                
                order=np.argsort(cls_probs)[::-1]
                bbox_order=cls_bbox_pred[order]
                
                pick=nms_cuda(bbox_order, nms_thresh=0.7, xyxy=True)
                pick=pick[:600]
                
                bboxes=bbox_order[pick].reshape(-1,bbox_order.shape[1])
                nms_scores=bboxes[:,-1]
                bboxes=bboxes[np.where(nms_scores>cfg.DET_SCORE_THRESH)[0],:]
                if bboxes.shape[0]>0:
                     print('Class {} has {} instances!'.format(CLASSES[i], bboxes.shape[0]))
                cls_bboxes.append(bboxes)

        ret['cls_bboxes']=cls_bboxes
        ret['proposals']=proposals_this_sample
        outputs.append(ret)
        
        start+=num_proposals_per_image[b]
    return outputs

def get_track_output(output_dict, configs):
    K=configs['K']
    temp_boxes=configs['temp_boxes']
    search_boxes=configs['search_boxes']
    rpn_conv_size=configs['rpn_conv_size']
    raw_anchors=configs['raw_anchors']
    bound=configs['bound']

    track_rpn_logits=output_dict['track_rpn_logits']
    track_rpn_bbox=output_dict['track_rpn_bbox']

    num_targets=track_rpn_logits.shape[0]

    track_rpn_cls=F.softmax(track_rpn_logits, dim=1).cpu().data.numpy()
    track_rpn_cls=track_rpn_cls[:,1,:,:].reshape(num_targets, K, rpn_conv_size, rpn_conv_size).transpose(0,2,3,1).reshape(num_targets, -1)
    track_rpn_bbox=track_rpn_bbox.cpu().data.numpy().transpose(0,2,3,1).reshape(num_targets, -1, 4)
    
    bboxes_list=[]
    anchors_list=[]

    for i in range(num_targets):
        temp_box=temp_boxes[i]
        temp_cx=0.5*(temp_box[0]+temp_box[2])
        temp_cy=0.5*(temp_box[1]+temp_box[3])

        rpn_cls=track_rpn_cls[i]
        rpn_bbox=track_rpn_bbox[i]
#        print(rpn_cls.size)
#        print(rpn_bbox.shape[0])
        order=np.argsort(rpn_cls)[::-1]
        
        _anchors=G.gen_region_anchors(raw_anchors, search_boxes[i].reshape(1,-1), bound, K, size=(rpn_conv_size, rpn_conv_size))[0]
        
        top=1
#        print(rpn_cls[order[:top]])
        
        fg_anchors=_anchors[order[:top]]
        fg_rpn_bbox=rpn_bbox[order[:top]]
                       
        if cfg.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            fg_rpn_bbox*=cfg.BBOX_STD_DEV
            
        bboxes=bbox_transform_inv(fg_anchors, fg_rpn_bbox)
        '''
        best_id=select_proposals_cosine(temp_boxes[i], bboxes, rpn_cls[order[:top]], order[:top], rpn_conv_size, K)
        anchors_list.append(fg_anchors[best_id][np.newaxis,:])
        bboxes_list.append(bboxes[best_id][np.newaxis,:])
        '''
        
        if np.max(rpn_cls[order[:top]])<cfg.TRACK_SCORE_THRESH:
            bboxes_list.append([])
        else:
            cx=0.5*(bboxes[:,0]+bboxes[:,2])
            cy=0.5*(bboxes[:,1]+bboxes[:,3])
            dist_cx=np.abs(cx-temp_cx)
            dist_cy=np.abs(cy-temp_cy)

            fg_inds=np.where(np.bitwise_and(dist_cx<cfg.TRACK_MAX_DIST, dist_cy<cfg.TRACK_MAX_DIST)==1)[0]
            bboxes_list.append(bboxes[fg_inds])
        
#        bboxes_list.append(bboxes)
        anchors_list.append(fg_anchors)    
    ret={}
    ret['bboxes_list']=bboxes_list
    ret['anchors_list']=anchors_list
    return ret           

def inference_detect(model, roidb, batch_size=-1):
    anchors=roidb['anchors']
    if isinstance(anchors, list):
        anchors=anchors[0]

    if cfg.IMAGE_NORMALIZE:
        roidb['temp_image']-=roidb['temp_image'].min()
        roidb['temp_image']/=roidb['temp_image'].max()
        roidb['det_image']-=roidb['det_image'].min()
        roidb['det_image']/=roidb['det_image'].max()
        roidb['temp_image']=(roidb['temp_image']-0.5)/0.5
        roidb['det_image']=(roidb['det_image']-0.5)/0.5
    else:
        roidb['temp_image']-=cfg.PIXEL_MEANS
        roidb['det_image']-=cfg.PIXEL_MEANS

    output_dict=model(roidb, task='det')    
    outputs=get_detect_output(output_dict, batch_size=batch_size)
    return outputs
    
def inference_track(model, roidb):
    rpn_conv_size=cfg.RPN_CONV_SIZE
    basic_size=cfg.TRACK_BASIC_SIZE
    ratios=cfg.TRACK_RATIOS
    scales=cfg.TRACK_SCALES
    K=len(ratios)*len(scales)
    
    raw_anchors=G.generate_anchors(basic_size, ratios, scales)
    
    if cfg.IMAGE_NORMALIZE:
        roidb['temp_image']-=roidb['temp_image'].min()
        roidb['temp_image']/=roidb['temp_image'].max()
        roidb['det_image']-=roidb['det_image'].min()
        roidb['det_image']/=roidb['det_image'].max()
        roidb['temp_image']=(roidb['temp_image']-0.5)/0.5
        roidb['det_image']=(roidb['det_image']-0.5)/0.5
    else:
        roidb['temp_image']-=cfg.PIXEL_MEANS
        roidb['det_image']-=cfg.PIXEL_MEANS
        
    bound=roidb['bound']
#    print(bound)
    
    output_dict=model(roidb, task='track')
    
    temp_boxes=roidb['temp_boxes']
    search_boxes=roidb['search_boxes']
    
    configs={}
    configs['K']=K
    configs['temp_boxes']=temp_boxes
    configs['search_boxes']=search_boxes
    configs['rpn_conv_size']=rpn_conv_size
    configs['raw_anchors']=raw_anchors
    configs['bound']=bound
    ret = get_track_output(output_dict, configs)
    bboxes_list=ret['bboxes_list']
    anchors_list=ret['anchors_list']
    return bboxes_list, anchors_list
    

def draw_detect_boxes(images, detects, with_proposals=True):
    assert len(images)==len(detects)
    
    for i in range(len(images)):
        image=images[i]
        ret=detects[i]
        cls_bboxes=ret['cls_bboxes']
        proposals=ret['proposals']
        
        for j in range(1,len(CLASSES)):
            bbox_with_score=cls_bboxes[j]
            if len(bbox_with_score)==0:
                continue
            
#            scores=bbox_with_score[:,-1]
#            fg_inds=np.where(scores>0.1)
#            bbox_with_score=bbox_with_score[fg_inds]
            x1,y1,x2,y2,scores=np.split(bbox_with_score, 5, axis=1)
            
            for k in range(bbox_with_score.shape[0]):
                cv2.rectangle(image, (x1[k],y1[k]),(x2[k],y2[k]), COLORS[j], 2)
                cv2.putText(image, '{}:{}'.format(CLASSES[j], scores[k]), (x1[k], y1[k]), cv2.FONT_HERSHEY_PLAIN, 0.9, COLORS[j], 1)
        
        if with_proposals:
            boxes=proposals[:,:4]
            for box in boxes:
                box=box.astype(np.int32)
                cv2.rectangle(image, (box[0],box[1]),(box[2],box[3]), (0,255,0), 1)

def draw_track_boxes(det_image, temp_boxes, search_boxes, bbox_list, anchor_list, fg_inds_local_list=None):
#    temp_image=images[0]
#    det_image=images[1]

    for i in range(temp_boxes.shape[0]):
        temp_box=temp_boxes[i].astype(np.int32)
        search_box=search_boxes[i].astype(np.int32)
        if len(bbox_list[i])==0:
            continue
        bboxes=bbox_list[i].astype(np.int32)
        fg_anchors=anchor_list[i].astype(np.int32)
        if fg_inds_local_list is not None:
            fg_inds=fg_inds_local_list[i]

#        cv2.rectangle(det_image, (search_box[0],search_box[1]), (search_box[2],search_box[3]), (0,255,255),2)
        cv2.rectangle(det_image, (temp_box[0],temp_box[1]), (temp_box[2],temp_box[3]), (255,0,0), 1)
        if fg_inds_local_list is not None:
            '''draw grids'''
            pass
#        for anchor in fg_anchors:
#            cv2.rectangle(det_image, (anchor[0], anchor[1]),(anchor[2],anchor[3]), (0,255,0), 1)
        for box in bboxes:
            cv2.rectangle(det_image, (box[0], box[1]),(box[2],box[3]), (0,255,0), 1)
        

if __name__=='__main__':
    from roidb.detrac_data_loader import DetracDataLoader
    
    cfg.PHASE='TEST'
    cfg.NUM_CLASSES=len(CLASSES)
    cfg.TEST.RPN_NMS_THRESH=0.6
    
    im_width=640
    im_height=384
    model_path='./ckpt/dl_mot_iter_640000.pkl'
    
    loader=DetracDataLoader(im_width, im_height, batch_size=1)
    
    model=MotFRCNN(im_width, im_height, pretrained=False)
    model.load_weights(model_path)
    model.cuda()
    
    roidbs=loader.get_minibatch()    
    if len(roidbs)==0:
        print('No targets, return')
    else:
        roidb=roidbs[0]
        images=[roidb['temp_image'].squeeze().astype(np.uint8, copy=True), roidb['det_image'].squeeze().astype(np.uint8, copy=True)]
        '''
        outputs=inference_detect(model ,roidb)
        draw_detect_boxes(images, outputs, with_proposals=False)
        for img in images:
            cv2.imshow('det', img)
            cv2.waitKey()
        '''
        bboxes_list, anchors_list=inference_track(model, roidb)
        det_image=images[1]
        draw_track_boxes(det_image, roidb['temp_boxes'], roidb['search_boxes'], bboxes_list, anchors_list, fg_inds_local_list=None)
        cv2.imwrite('result.jpg', det_image)
        cv2.imshow('det', det_image)
        print('Track result has been written to result.jpg')
        cv2.waitKey()     
        
            