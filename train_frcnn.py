import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.variable import Variable as Variable
from mynn import DataParallel

import sys
sys.path.insert(0, 'E:/yufji/DLMOT')
import os
import numpy as np
import cv2

from model.mot_frcnn import MotFRCNN
from roidb.data_loader import DataLoader
from roidb.mot_data_loader import MOTDataLoader
from roidb.vid_data_loader import VIDDataLoader
from roidb.detrac_data_reader import DetracDataReader
import rpn.generate_anchors as G
import rpn.anchor_target as T
import rpn.util as U
from core.config import cfg

from time import time

import logging

logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(levelname)s - %(message)s')

BBOX_OVERLAP_PRECOMPUTED=False
RESOLUTION={'VGG':(768,448),'RESNET':(640,384),'ALEX':(399,255)}
STRIDE={'VGG':8,'RESNET':8,'ALEX':8}
net_type='RESNET'

num_gpus=1
gpu_ids=0

DEBUG=False

def set_gpu_id():
    global num_gpus, gpu_ids
    gpu_id=cfg.GPU_ID
    if isinstance(gpu_id, list):
        num_gpus=len(gpu_id)
        gpu_id=','.join(list(map(str, gpu_id)))
    elif isinstance(gpu_id, int):
        gpu_id=str(gpu_id)
    else:
        raise EnvironmentError
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_id
    print('Using GPUs {}'.format(gpu_id))
    gpu_ids=list(range(len(cfg.GPU_ID)))

class TrainEngine(object):
    def __init__(self, backbone_pretrained=False):
        self.batch_size=2
        self.stride=STRIDE[net_type]   
        self.im_w, self.im_h=RESOLUTION[net_type]

        self.display=20
        self.snapshot=2
        self.decay_ratio=0.1
        
        self.lr_mult=0.5
        
        self.fetch_config()
        self.update_config()
        
        self.K=len(self.ratios)*len(self.scales)
        self.TK=len(self.track_ratios)*len(self.track_scales)                    

        self.raw_anchors=G.generate_anchors(self.basic_size, self.ratios, self.scales)
        self.track_raw_anchors=G.generate_anchors(self.track_basic_size, self.track_ratios, self.track_scales)
        
        self.backbone_pretrained=backbone_pretrained

        self.model=MotFRCNN(self.im_w, self.im_h, pretrained=self.backbone_pretrained)

    def update_config(self):
        cfg[cfg.PHASE].IMS_PER_BATCH=self.batch_size
        cfg.DATA_LOADER_METHOD='INTER_IMG'        
        cfg.STRIDE=self.stride
        cfg.TEMP_MIN_SIZE=64
        cfg.TEMP_MAX_SIZE=min(self.im_h, self.im_w)
        cfg.TEMP_NUM=4
        cfg.GAIN=0.012

    def fetch_config(self):
        self.basic_size=cfg.BASIC_SIZE
        self.ratios=cfg.RATIOS
        self.scales=cfg.SCALES

        self.track_basic_size=cfg.TRACK_BASIC_SIZE
        self.track_ratios=cfg.TRACK_RATIOS
        self.track_scales=cfg.TRACK_SCALES
        self.rpn_conv_size=cfg.RPN_CONV_SIZE 

    def xy2wh(self, boxes):
        x=boxes[:,0]
        y=boxes[:,1]
        w=boxes[:,2]-x+1
        h=boxes[:,3]-y+1
        return x,y,w,h
     
    def vis_anchors(self, image, temp_boxes, det_boxes, search_boxes, anchors, fg_anchor_inds):
        '''
        fg_anchors_inds: all inds in a frame, not a box
        '''
        temp_box=temp_boxes.astype(np.int32).squeeze()
        det_box=det_boxes.astype(np.int32).squeeze()
        search_box=search_boxes.astype(np.int32).squeeze()

        cv2.rectangle(image, (temp_box[0],temp_box[1]),(temp_box[2],temp_box[3]),(255,0,0),1)
        cv2.rectangle(image, (det_box[0],det_box[1]),(det_box[2],det_box[3]), (0,0,255), 2)
        cv2.rectangle(image, (search_box[0],search_box[1]),(search_box[2],search_box[3]), (0,255,255), 2)
        
        anchors_all=np.vstack(anchors).astype(np.int32)
        for fg_ind in fg_anchor_inds:
            anchor=anchors_all[fg_ind].astype(np.int32)
            ctrx=(anchor[0]+anchor[2])//2
            ctry=(anchor[1]+anchor[3])//2
            cv2.circle(image, (ctrx,ctry), 2, (255,0,0), -1)
            cv2.rectangle(image, (anchor[0],anchor[1]),(anchor[2],anchor[3]), (0,255,0), 1)
            
    def get_param_groups(self, base_lr):
        backbone_lr=base_lr
        backbone_lr*=self.lr_mult
        return self.model.get_params({'backbone':backbone_lr, 'task':base_lr})
    
    def get_optimizer(self, lr):
        if self.backbone_pretrained:
            params=self.get_param_groups(lr)
            if cfg.OPTIMIZER=='sgd':
                optimizer=optim.SGD(params, lr=lr, momentum=0.01)
            elif cfg.OPTIMIZER=='adam':
                optimizer=optim.Adam(params, lr=lr)
            else:
                raise NotImplementedError
            for param_group in optimizer.param_groups:
                print('{} has learning rate {}'.format(param_group['key'], param_group['lr']))
        else:
            if cfg.OPTIMIZER=='sgd':
                optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.01) 
            elif cfg.OPTIMIZER=='adam':
                optimizer = optim.Adam(self.model.parameters(), lr=lr) 
            else:
                raise NotImplementedError
        return optimizer

    def save_ckpt(self, stepvalues, epoch, lr, logger):
        model_name='ckpt/dl_mot_epoch_{}.pkl'.format(int(epoch))    
#        model_name='ckpt_mot/dl_mot_{}_{}.pkl'.format(self.snapshot_type,int(n))
        msg='Snapshotting to {}'.format(model_name)
        logger.info(msg)
        model={'model':self.model.state_dict(),'epoch':epoch, 'lr':lr, 'stepvalues':stepvalues, 'im_w':self.im_w, 'im_h':self.im_h}
        torch.save(model, model_name)
    
    def restore_from_ckpt(self, ckpt_path, logger):
        ckpt=torch.load(ckpt_path)
#        model=ckpt['model']
        epoch=ckpt['epoch']+1
        lr=ckpt['lr']
        stepvalues=ckpt['stepvalues']
        msg='Restoring from {}\nStart epoch: {}, current lr: {}'.format(ckpt_path,\
              epoch, lr)
        logger.info(msg)
#        self.model.update_state_dict(model)
        logger.info('Model update successfully')
        return stepvalues, epoch, lr
    
    def restore_from_epoch(self, start_epoch, stepvalues, lr, iters_per_epoch):
        epoch_iters=int(start_epoch*iters_per_epoch)
        base_lr=lr
        stepvalues_new=[]
        for i in stepvalues:
            if i>start_epoch:
                stepvalues_new.append(i)
            else:
                lr*=self.decay_ratio
        print('lr drops from {} to {}'.format(base_lr, lr))
        print('Restoring from epoch {}, iters {}'.format(start_epoch, epoch_iters))
        stepvalues=stepvalues_new
        return stepvalues, epoch_iters, lr
        
    def train(self, pretrained_model=None):
        dataset='detrac'
        reader=None
        if dataset=='detrac':
            reader=DetracDataReader(self.im_w, self.im_h, batch_size=self.batch_size)
        else:
            raise NotImplementedError

        logger = logging.getLogger(__name__)

        backbone_pretrained=True
        
        num_samples=reader.__len__()
        
        logger.info('Load {} samples'.format(num_samples))
        
        num_epochs=60
        lr=0.00012
        stepvalues = [40]

        if pretrained_model is None:
            self.model.init_weights()
        else:
            self.model.load_weights(model_path=pretrained_model)        
        
        self.model.cuda(gpu_ids[0])
        
        num_vis_anchors=100        
        num_jumped=0

        iters_per_epoch=int(num_samples/self.batch_size)
        start_epoch=0
        
        if pretrained_model is not None:
            stepvalues, start_epoch, lr=self.restore_from_ckpt(pretrained_model, logger)
        else:
            stepvalues, epoch_iters, lr=self.restore_from_epoch(start_epoch, stepvalues, lr, iters_per_epoch)
        
        epoch_iters=iters_per_epoch*start_epoch
            
        optimizer=self.get_optimizer(lr)

        logger.info('Start training...')
        
        if num_gpus>1:
            self.model=DataParallel(self.model, device_ids=gpu_ids)
        
        data_loader=DataLoader(reader, shuffle=True, batch_size=self.batch_size, num_workers=self.batch_size)

        for epoch in range(start_epoch, num_epochs):
            
            for iters, roidbs in enumerate(data_loader):
                B=len(roidbs)
                if B==0:
                    msg='No reference image in this minibatch, jump!'
                    logger.info(msg)
                    num_jumped+=1
                else:
                    '''NHWC'''
                    vis_image_list=[]                    
                    for db in roidbs:
                        if epoch==start_epoch and iters<num_vis_anchors:
                            vis_image_list.append(db['det_image'].squeeze(0).astype(np.uint8, copy=True))
                        
                        if cfg.IMAGE_NORMALIZE:
                            db['temp_image'] /= 255.0
                            db['det_image'] /= 255.0
                        else:
                            db['temp_image'] -= cfg.PIXEL_MEANS
                            db['det_image'] -= cfg.PIXEL_MEANS

                    tic=time()
                    output_dict=self.model(roidbs)

                    toc=time()
                    if DEBUG:
                        print('Forward costs {}s'.format(toc-tic))

                    det_anchors=roidbs[0]['anchors']

                    temp_boxes_siam=output_dict['temp_boxes4siamese']
                    det_boxes_siam=output_dict['det_boxes4siamese']
                    search_boxes_siam=output_dict['search_boxes4siamese']

                    track_anchors=output_dict['track_anchors']

                    bound=(self.im_w, self.im_h)
                    out_size=(self.im_w//self.stride, self.im_h//self.stride)

                    num_boxes_track=np.ones(len(track_anchors), dtype=np.int32).tolist()
                    
#                    track_anchors=G.gen_region_anchors(self.track_raw_anchors, search_box.reshape(1,-1), bound, K=self.TK, size=(rpn_conv_size, rpn_conv_size))[0]
                    track_anchor_cls_targets, track_anchor_bbox_targets, track_bbox_weights, track_fg_anchor_inds=\
                        T.compute_track_rpn_targets(track_anchors, det_boxes_siam, \
                            bbox_overlaps=None, bound=bound, rpn_conv_size=self.rpn_conv_size,K=self.TK)
                    
                    gt_boxes=output_dict['gt_boxes']

                    detect_anchor_cls_targets, detect_anchor_bbox_targets, detect_bbox_weights, detect_fg_anchor_inds=\
                        T.compute_detection_rpn_targets(det_anchors, gt_boxes, out_size, \
                             bbox_overlaps=None, K=self.K, batch_size=2*self.batch_size)

                    track_anchor_cls_targets=Variable(torch.from_numpy(track_anchor_cls_targets).long().cuda(async=True))
                    track_anchor_bbox_targets=Variable(torch.from_numpy(track_anchor_bbox_targets).float().cuda(async=True))                   

                    track_bbox_weights_var=Variable(torch.from_numpy(track_bbox_weights).float().cuda(async=True), requires_grad=False)
                    track_rpn_bbox=torch.mul(output_dict['track_rpn_bbox'], track_bbox_weights_var)
                    track_anchor_bbox_targets=torch.mul(track_anchor_bbox_targets, track_bbox_weights_var)

                    detect_anchor_cls_targets=Variable(torch.from_numpy(detect_anchor_cls_targets).long().cuda(async=True))
                    detect_anchor_bbox_targets=Variable(torch.from_numpy(detect_anchor_bbox_targets).float().cuda(async=True))                   

                    detect_bbox_weights_var=Variable(torch.from_numpy(detect_bbox_weights).float().cuda(async=True), requires_grad=False)
                    detect_rpn_bbox=torch.mul(output_dict['detect_rpn_bbox'], detect_bbox_weights_var)
                    detect_anchor_bbox_targets=torch.mul(detect_anchor_bbox_targets, detect_bbox_weights_var)

                    if epoch==start_epoch and iters<num_vis_anchors:
                        start=0
                        for i, canvas in enumerate(vis_image_list):
                            left=start
                            right=start+num_boxes_track[i]
                            canvas_cpy=canvas.copy()
                            self.vis_anchors(canvas_cpy, temp_boxes_siam[left:right], det_boxes_siam[left:right], search_boxes_siam[left:right], track_anchors[left:right], track_fg_anchor_inds[i])
                            cv2.imwrite('vis_anchors/vis_{}.jpg'.format(iters), canvas_cpy)
                            start+=num_boxes_track[i]
                    
                    '''
                    In CrossEntropyLoss, input is BCHW, target is BHW, NOT BCHW!!! 
                    '''
                    track_num_examples=0
                    detect_num_examples=0
                    for fg_inds in track_fg_anchor_inds:
                        track_num_examples+=len(fg_inds)
                    for fg_inds in detect_fg_anchor_inds:
                        detect_num_examples+=len(fg_inds)
                        
                    num_fg_proposals=output_dict['num_fgs']
                    
                    track_denominator_rpn=track_num_examples if track_num_examples>0 else 1
                    detect_denominator_rpn=detect_num_examples if detect_num_examples>0 else 1

                    denominator_frcnn=num_fg_proposals if num_fg_proposals>0 else 1
                    track_denominator_rpn+=1e-4
                    denominator_frcnn+=1e-4

                    gain=1.0
                    track_rpn_loss_cls=F.cross_entropy(output_dict['track_rpn_logits'], track_anchor_cls_targets, size_average=True, ignore_index=-100)
                    track_rpn_loss_bbox=F.smooth_l1_loss(track_rpn_bbox, track_anchor_bbox_targets, size_average=False, reduce=False)
                    detect_rpn_loss_cls=F.cross_entropy(output_dict['detect_rpn_logits'], detect_anchor_cls_targets, size_average=True, ignore_index=-100)
                    detect_rpn_loss_bbox=F.smooth_l1_loss(detect_rpn_bbox, detect_anchor_bbox_targets, size_average=False, reduce=False)

                    frcnn_loss_cls=F.cross_entropy(output_dict['frcnn_logits'], output_dict['frcnn_cls_target'])
                    frcnn_loss_bbox=F.smooth_l1_loss(output_dict['frcnn_bbox'], output_dict['frcnn_bbox_target'], size_average=False, reduce=False)
                    
                    track_rpn_loss_bbox=gain*torch.div(torch.sum(track_rpn_loss_bbox, dim=1), 4.0)
                    track_rpn_loss_bbox=gain*torch.div(torch.sum(track_rpn_loss_bbox), track_denominator_rpn)
                    detect_rpn_loss_bbox=gain*torch.div(torch.sum(detect_rpn_loss_bbox, dim=1), 4.0)
                    detect_rpn_loss_bbox=gain*torch.div(torch.sum(detect_rpn_loss_bbox), detect_denominator_rpn)
                    
                    frcnn_loss_bbox=gain*torch.div(torch.sum(frcnn_loss_bbox, dim=1), 4.0)
                    frcnn_loss_bbox=gain*torch.div(torch.sum(frcnn_loss_bbox), denominator_frcnn)  
                    
                    '''Do NOT multiply margin in RPN'''                    
                    loss=track_rpn_loss_cls+track_rpn_loss_bbox+detect_rpn_loss_cls+detect_rpn_loss_bbox+frcnn_loss_cls+frcnn_loss_bbox
#                    print(loss.device)
                    
                    if iters%self.display==0:
                        msg='Epoch {}/{}. Iter_epoch {}/{}. Global_iter: {}. Loss: {}. track_rpn_loss_cls: {}. track_rpn_loss_bbox: {}. detect_rpn_loss_cls: {}. detect_rpn_loss_bbox: {}. frcnn_loss_cls: {}. frcnn_loss_bbox: {}. lr: {}. track_num_examples: {}. detect_num_examples: {}. num_proposals: {}.'.format(epoch, num_epochs, iters, iters_per_epoch, epoch_iters, \
                                        loss.item(), track_rpn_loss_cls.item(), track_rpn_loss_bbox.item(), \
                                        detect_rpn_loss_cls.item(), detect_rpn_loss_bbox.item(), \
                                        frcnn_loss_cls.item(), frcnn_loss_bbox.item(), \
                                        lr, track_num_examples, detect_num_examples, num_fg_proposals)
                        
                        logger.info(msg)

                    loss_val=loss.cpu().data.numpy()
                        
                    if loss_val > 1e7:
                        msg='Loss too large, stop! {}.'.format(loss_val)
                        logger.error(msg)
                        assert 0
                    if np.isnan(loss_val):
                        msg='Loss nan, stop! {}.'.format(loss_val)
                        logger.error(msg)
                        assert 0
                    tic=time()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    toc=time()
                    if DEBUG:
                        print('Backward costs {}s'.format(toc-tic))

                epoch_iters+=1
                
            if epoch>0 and epoch%self.snapshot==0:
                self.save_ckpt(stepvalues, epoch, lr, logger)
                
            if len(stepvalues)>0 and (epoch+1)==stepvalues[0]:
                lr*=self.decay_ratio
                msg='learning rate decay: %e'%lr
                logger.info(msg)
                for param_group in optimizer.param_groups:
                    param_group['lr']=lr
                    if backbone_pretrained and param_group['key']=='backbone':
                        param_group['lr']*=self.lr_mult
                stepvalues.pop(0)                
            
        self.save_ckpt(stepvalues, epoch, lr, logger)
        
        msg='Finish training!!\nTotal jumped batch: {}'.format(num_jumped)
        logger.info(msg)
            
if __name__=='__main__':
    set_gpu_id()

    cfg.PHASE='TRAIN'
    cfg.TRAIN.RPN_NMS_THRESH=0.7
    cfg.NUM_CLASSES=5
#    pretrained='./dl_mot_pretrained.pkl'
    pretrained='./ckpt/dl_mot_epoch_12.pkl'
    engine=TrainEngine(backbone_pretrained=True)
    #engine.train(pretrained_model=pretrained)
    engine.train()
