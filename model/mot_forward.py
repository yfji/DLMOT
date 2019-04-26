'''
Siamese-RPN

@author: yfji

2018.9.1.21
'''
'''
For forward only
Multi-object
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable

import math
import numpy as np
from collections import OrderedDict
from model.vgg16 import Vgg16
import model.resnet as resnet
import model.fpn as fpn
import model.myrpn as rpn
from model.my_fast_rcnn import FastRCNN
import rpn.util as U
#from rpn.nms import nms
from fast_rcnn.proposal_target import get_proposal_target
from core.config import cfg

from nms.nms_wrapper import nms
#from roialign.roi_align.crop_and_resize import CropAndResizeFunction
from roi_align.functions.roi_align import RoIAlignFunction

DEBUG=False

# def crop_and_resize(pool_size, feature_map, boxes, box_ind):
#     if boxes.shape[1]==5:
#         x1, y1, x2, y2, _= boxes.chunk(5, dim=1)
#     else:
#         x1, y1, x2, y2= boxes.chunk(4, dim=1)
#     im_h, im_w=feature_map.shape[2:4]
#     x1=x1/(float(im_w-1))
#     x2=x2/(float(im_w-1))
#     y1=y1/(float(im_h-1))
#     y2=y2/(float(im_h-1))

#     boxes = torch.cat((y1, x1, y2, x2), 1)
#     return CropAndResizeFunction(pool_size[0],pool_size[1],0)(feature_map, boxes, box_ind)
# Windows version included
def crop_and_resize(pool_size, feature_map, boxes, box_ind):
    if boxes.shape[1]==5:
        x1, y1, x2, y2, _= boxes.chunk(5, dim=1)
    else:
        x1, y1, x2, y2= boxes.chunk(4, dim=1)
    box_ind=box_ind.view(-1,1).float()
    boxes = torch.cat((box_ind, x1, y1, x2, y2), 1)
    return RoIAlignFunction(pool_size[0],pool_size[1], 1)(feature_map, boxes)

def nms_cuda(boxes_np, nms_thresh=0.7, xyxy=True):    
    if xyxy:
        x1,y1,x2,y2,scores=np.split(boxes_np, 5, axis=1)
        boxes_np=np.hstack([y1,x1,y2,x2,scores])
    boxes_pth=torch.from_numpy(boxes_np).float().cuda()
    pick=nms(boxes_pth, nms_thresh)
    pick=pick.cpu().data.numpy()
#    print(pick)
    if len(pick.shape)==2:
        pick=pick.squeeze(1)
    return pick

class MotFRCNN(nn.Module):
    def __init__(self, im_width, im_height, pretrained=True):
        super(MotFRCNN, self).__init__()
        
        self.rpn_out_ch=512
        self.track_rpn_out_ch=256
        self.features_out_ch=256

        self.fetch_config()

        self.bound=(im_width, im_height)
        self.out_size=(im_width//self.stride, im_height//self.stride)

        self.num_anchors=self.K*(self.rpn_conv_size**2)

        self.fpn=fpn.FPN(self.features_out_ch)
        self.features=resnet.resnet50(pretrained=pretrained)
        self.make_rpn()
        
    def fetch_config(self):
        self.det_roi_size=cfg.DET_ROI_SIZE
        self.temp_roi_size=cfg.TEMP_ROI_SIZE
        
        self.frcnn_roi_size=cfg.FRCNN_ROI_SIZE
        self.stride=cfg.STRIDE
        self.K=len(cfg.RATIOS)*len(cfg.SCALES)
        self.TK=len(cfg.TRACK_RATIOS)*len(cfg.TRACK_SCALES)

        self.use_bn=not cfg.IMAGE_NORMALIZE
        self.rpn_conv_size=cfg.RPN_CONV_SIZE
        
    def load_weights(self, model_path=None):
        print('loading model from {}'.format(model_path))
        ckpt = torch.load(model_path)
        keys=list(ckpt.keys())
        if 'epoch' in keys:
            print('Restoring model from self-defined ckpt')
            im_w, im_h=ckpt['im_w'], ckpt['im_h']
            print('Using resolution: {}x{}'.format(im_w,im_h))
            model=ckpt['model']
            cfg.TEMP_MIN_SIZE=ckpt['temp_min_size']
            cfg.TEMP_MAX_SIZE=ckpt['temp_max_size']
            cfg.TEMP_NUM=ckpt['temp_num']
            print('Using template config: {},{},{}'.format(cfg.TEMP_MIN_SIZE,cfg.TEMP_MAX_SIZE,cfg.TEMP_NUM))
            self.load_state_dict(model)
        print('Load model successfully')

    def make_rpn(self):
        self.rpn_out_ch=self.features.out_ch

        self.track_rpn=rpn.RPN(self.features_out_ch, self.track_rpn_out_ch, 2*self.TK*self.track_rpn_out_ch, 4*self.TK*self.track_rpn_out_ch)
        self.detect_rpn=rpn.RPN(self.features_out_ch, self.rpn_out_ch, 2*self.K, 4*self.K)

        self.det_cls_conv=nn.Conv2d(self.features_out_ch, self.track_rpn_out_ch, 1, 1, padding=0)
        self.det_bbox_conv=nn.Conv2d(self.features_out_ch, self.track_rpn_out_ch, 1, 1, padding=0)
        
        self.fastRCNN=FastRCNN(depth=self.features_out_ch, pool_size=self.frcnn_roi_size, num_classes=cfg.NUM_CLASSES)
        self.rpn_cls_softmax=nn.Softmax(dim=1)
        
    def clip_boxes(self, boxes, bound):
        boxes[:,0]=np.minimum(bound[0],np.maximum(0, boxes[:,0]))
        boxes[:,1]=np.minimum(bound[1],np.maximum(0, boxes[:,1]))
        boxes[:,2]=np.maximum(0,np.minimum(bound[0], boxes[:,2]))
        boxes[:,3] = np.maximum(0, np.minimum(bound[1], boxes[:,3]))

    def gen_proposals(self, rpn_cls, rpn_bbox, anchors, batch_size):
#        out_cls_softmax=self.rpn_cls_softmax(rpn_cls)
        out_cls_softmax=F.softmax(rpn_cls, dim=1)
        out_cls_np=out_cls_softmax.cpu().data.numpy()[:,1,:,:]
        out_cls_np=out_cls_np.reshape(batch_size, self.K, self.out_size[1], self.out_size[0]).transpose(0,2,3,1).reshape(-1, 1)
        out_bbox_np=rpn_bbox.cpu().data.numpy().transpose(0,2,3,1).reshape(-1, 4)
        
        if cfg.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            out_bbox_np = out_bbox_np*cfg.RPN_BBOX_STD_DEV

        if anchors.shape[0]!=out_bbox_np.shape[0]:
            anchors=np.tile(anchors, (batch_size,1))

        bbox_pred=U.bbox_transform_inv(anchors, out_bbox_np)
        self.clip_boxes(bbox_pred, self.bound)
        
        bbox_pred_with_cls=np.hstack((bbox_pred, out_cls_np))
        proposals_batch=np.split(bbox_pred_with_cls, batch_size, axis=0)
        
        all_proposals=[]
       
        for proposals in proposals_batch:
            if cfg.NMS:
                order=np.argsort(proposals[:,-1])[::-1]
                proposals_order=proposals[order[:cfg[cfg.PHASE].RPN_PRE_NMS_TOP_N]]
                pick=nms_cuda(proposals_order, nms_thresh=cfg[cfg.PHASE].RPN_NMS_THRESH, xyxy=True)
                
                if len(pick)==0:
                    print('No pick in proposal nms')
                if cfg[cfg.PHASE].RPN_POST_NMS_TOP_N>0 and len(pick)>cfg[cfg.PHASE].RPN_POST_NMS_TOP_N:
                    pick=pick[:cfg[cfg.PHASE].RPN_POST_NMS_TOP_N]
                proposals_nms=proposals_order[pick] 
                all_proposals.append(proposals_nms)               
            else:
                all_proposals.append(proposals)
    
        '''list of all proposals of each box, not each frame'''
        return all_proposals
    
    def get_rois(self, featuremap, proposals, num_proposals):
        proposals=proposals/(1.0*self.stride)
        box_inds_np=[i*np.ones(num_proposals[i], dtype=np.int32) for i in range(len(num_proposals))]
        box_inds_np=np.concatenate(box_inds_np)
        box_inds=Variable(torch.from_numpy(box_inds_np).cuda())
        '''list'''
        if isinstance(proposals, list):
            proposals_all=np.vstack(proposals)
        else:
            proposals_all=proposals
        roi_features=crop_and_resize((self.frcnn_roi_size, self.frcnn_roi_size),featuremap, Variable(torch.from_numpy(proposals_all).float().cuda()), box_inds)
        return roi_features

    def fpn_out(self, z_input, x_input):
        z_c1, z_c2, z_c3, z_c4, z_c5=self.features(z_input)
        x_c1, x_c2, x_c3, x_c4, x_c5=self.features(x_input)

        z_p2,z_p3,_,_,_=self.fpn(z_c1,z_c2,z_c3,z_c4,z_c5)
        x_p2,x_p3,_,_,_=self.fpn(x_c1,x_c2,x_c3,x_c4,x_c5)

        if self.stride==8:
            return z_p3, x_p3
        else:
            return z_p2, x_p2
    
    def forward_track(self, roidb, x, z):
        temp_boxes=roidb['temp_boxes']     
        search_boxes=roidb['search_boxes']        

        num_boxes=[temp_boxes.shape[0]]
        n_boxes=sum(num_boxes)        

        self.temp_boxes=temp_boxes
        self.search_boxes=search_boxes
        self.num_boxes=num_boxes

        temp_boxes = temp_boxes/(1.0*self.stride)
        search_boxes = search_boxes/(1.0*self.stride)
        
        bound2=(x.shape[3], x.shape[2])
        self.clip_boxes(temp_boxes, bound2)
        self.clip_boxes(search_boxes, bound2)
        
        '''siamese rpn begin'''
        
        box_inds=torch.zeros(n_boxes, dtype=torch.int32).cuda()

        temp_boxes=torch.from_numpy(temp_boxes.astype(np.float32)).cuda()
        search_boxes=torch.from_numpy(search_boxes.astype(np.float32)).cuda()

        temp_box_roi_feat=crop_and_resize((self.temp_roi_size,self.temp_roi_size), z, Variable(temp_boxes), Variable(box_inds))
        det_box_roi_feat=crop_and_resize((self.det_roi_size,self.det_roi_size), x, Variable(search_boxes), Variable(box_inds))
        
        if DEBUG:
            print('temp roi feature map shape: ', end='');print(temp_box_roi_feat.shape)
            print('det roi feature map shape: ', end='');print(det_box_roi_feat.shape)

        track_rpn_cls_temp, track_rpn_bbox_temp=self.track_rpn(temp_box_roi_feat)        
        track_rpn_cls_det=self.det_cls_conv(det_box_roi_feat)
        track_rpn_bbox_det=self.det_bbox_conv(det_box_roi_feat)
          
        if DEBUG:
            print('track_rpn_cls_temp shape: ', end='');print(track_rpn_cls_temp.shape)
            print('track_rpn_bbox_temp shape: ', end='');print(track_rpn_bbox_temp.shape)
            print('track_rpn_cls_det shape: ', end='');print(track_rpn_cls_det.shape)
            print('track_rpn_bbox_det shape: ', end='');print(track_rpn_bbox_det.shape)

        temp_ksize=self.temp_roi_size
        track_rpn_cls_temp=track_rpn_cls_temp.view(2*self.TK*n_boxes, self.track_rpn_out_ch, temp_ksize, temp_ksize)
        track_rpn_bbox_temp=track_rpn_bbox_temp.view(4*self.TK*n_boxes, self.track_rpn_out_ch, temp_ksize, temp_ksize)
        
        track_rpn_logits=F.conv2d(track_rpn_cls_det, track_rpn_cls_temp)    #[N,2KN,h',w']
        track_rpn_bbox=F.conv2d(track_rpn_bbox_det, track_rpn_bbox_temp)    #[N,4KN,h',w']

        if DEBUG:
            print('track_rpn_logits shape: ',end='');print(track_rpn_logits.shape)    
            print('track_rpn_bbox shape: ',end='');print(track_rpn_bbox.shape)

        track_rpn_logits=track_rpn_logits.view(n_boxes**2, 2*self.TK, self.rpn_conv_size, self.rpn_conv_size)  #[N,N,2K,h',w']
        track_rpn_bbox=track_rpn_bbox.view(n_boxes**2, 4*self.TK, self.rpn_conv_size, self.rpn_conv_size)  #[N,N,4K,h',w']

        inds=torch.from_numpy(np.linspace(0,n_boxes**2-1,n_boxes)).long()
        track_rpn_logits=track_rpn_logits[inds].view(n_boxes,2,self.TK*self.rpn_conv_size,self.rpn_conv_size)
        track_rpn_bbox=track_rpn_bbox[inds].view(n_boxes,4*self.TK,self.rpn_conv_size,self.rpn_conv_size)
        
#        track_rpn_logits=track_rpn_logits.view(n_boxes,2,self.TK*self.rpn_conv_size,self.rpn_conv_size)
        if DEBUG:
            print('track_rpn_logits shape: ',end='');print(track_rpn_logits.shape)    
#        track_rpn_logits=track_rpn_logits.view(1, 2, self.rpn_conv_size, self.rpn_conv_size)
        '''siamese rpn end'''
        ret={}
        ret['track_rpn_logits']=track_rpn_logits
        ret['track_rpn_bbox']=track_rpn_bbox
        return ret

    def forward_detect(self, roidb, batch_x):
        '''Fast RCNN start'''
        #[2,2K,H,W]
        #[2,4K,H,W]
        '''anchors the same for temp and det'''
        anchors=roidb['anchors']    
        detect_rpn_logits, detect_rpn_bbox=self.detect_rpn(batch_x)        
        detect_rpn_logits=detect_rpn_logits.view(2, 2, self.K*self.out_size[1], self.out_size[0])
        
        all_proposals=self.gen_proposals(detect_rpn_logits, detect_rpn_bbox, anchors, 2)
                
        num_proposals_per_box=[]
        '''
        if nms, proposals may not have the same number of each box
        if not nms, each box has same number of proposals
        '''
        for prop in all_proposals:
            num_proposals_per_box.append(prop.shape[0])
        proposals=np.vstack(all_proposals)
        roi_features=self.get_rois(batch_x, proposals, [len(proposals)])
        frcnn_logits, frcnn_probs, frcnn_bbox=self.fastRCNN(roi_features)
        
        '''Fast-RCNN end'''
        ret={}
        ret['detect_rpn_logits']=detect_rpn_logits
        ret['detect_rpn_bbox']=detect_rpn_bbox
        ret['frcnn_logits']=frcnn_logits
        ret['frcnn_probs']=frcnn_probs
        ret['frcnn_bbox']=frcnn_bbox
        ret['proposals']=proposals
        ret['num_proposals']=num_proposals_per_box
        return ret

    '''Single object'''
#    def forward(self, temp_image, det_image, temp_box, search_box):
    def forward(self, roidb, task='all'):
        temp_image=roidb['temp_image']
        det_image=roidb['det_image']                      
        
        #NHWC-->NCHW
        temp_image=Variable(torch.from_numpy(temp_image.transpose(0, 3, 1, 2)).cuda())
        det_image=Variable(torch.from_numpy(det_image.transpose(0, 3, 1, 2)).cuda())
        
        z,x=self.fpn_out(temp_image, det_image)

        batch_x=torch.cat([z,  x], 0)
        if DEBUG:
            print('Batch feature map shape: ', end='');print(batch_x.shape)
        
        output={}
        if task=='all':
#            track_rpn_logits, track_rpn_bbox=forward_track(roidb, x,z)
#            detect_rpn_logits, detect_rpn_bbox, frcnn_logits, frcnn_probs, frcnn_bbox=forward_detect(roidb, batch_x)
            ret_track=self.forward_track(roidb, x, z)
            ret_det=self.forward_detect(roidb, batch_x)
            
            output['detect_rpn_logits']=ret_det['detect_rpn_logits']
            output['detect_rpn_bbox']=ret_det['detect_rpn_bbox']
            output['frcnn_bbox']=ret_det['frcnn_bbox']
            output['frcnn_probs']=ret_det['frcnn_probs']
            output['proposals']=ret_det['proposals']
            output['num_proposals']=ret_det['num_proposals']
            
            output['track_rpn_logits']=ret_track['track_rpn_logits']
            output['track_rpn_bbox']=ret_track['track_rpn_bbox']

        elif task=='det':
            ret_det=self.forward_detect(roidb, batch_x)            
            output['detect_rpn_logits']=ret_det['detect_rpn_logits']
            output['detect_rpn_bbox']=ret_det['detect_rpn_bbox']
            output['frcnn_bbox']=ret_det['frcnn_bbox']
            output['frcnn_probs']=ret_det['frcnn_probs']
            output['proposals']=ret_det['proposals']
            output['num_proposals']=ret_det['num_proposals']
#        elif task=='track':
        else:
            ret_track=self.forward_track(roidb, x, z)
            output['track_rpn_logits']=ret_track['track_rpn_logits']
            output['track_rpn_bbox']=ret_track['track_rpn_bbox']
        

#        return out_cls, out_bbox, all_proposals
        return output  

if __name__=='__main__':
    model=MotFRCNN(100,100)
#    torch.save(model.state_dict(), 'siamRPN.pkl')
    params=model.get_params({'backbone':0.1, 'task':0.5})