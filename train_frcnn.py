import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.variable import Variable as Variable

import os
import numpy as np
import cv2

from model.mot_frcnn import MotFRCNN, VGG_PRETRAINED, VGG_PRETRAINED_BN
from roidb.data_loader import DataLoader
from roidb.mot_data_loader import MOTDataLoader
#from roidb.vid_data_loader import VIDDataLoader
from roidb.vid_data_loader import VIDDataLoader
from roidb.detrac_data_loader import DetracDataLoader
import rpn.generate_anchors as G
import rpn.anchor_target as T
import rpn.util as U
from core.config import cfg

import logging

logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(levelname)s - %(message)s')

BBOX_OVERLAP_PRECOMPUTED=False
RESOLUTION={'VGG':(768,448),'RESNET':(768,448),'ALEX':(399,255)}
STRIDE={'VGG':8,'RESNET':8,'ALEX':8}
net_type='RESNET'

class TrainEngine(object):
    def __init__(self):
        self.batch_size=2
        
        self.stride=STRIDE[net_type]   
        self.im_w, self.im_h=RESOLUTION[net_type]

        self.display=20
        self.snapshot=20000
        self.decay_ratio=0.1
        
        self.lr_mult=0.5
        
        self.fetch_config()
        self.update_config()
        
        self.K=len(self.ratios)*len(self.scales)
        self.TK=len(self.track_ratios)*len(self.track_scales)                    

        self.raw_anchors=G.generate_anchors(self.basic_size, self.ratios, self.scales)
        self.track_raw_anchors=G.generate_anchors(self.track_basic_size, self.track_ratios, self.track_scales)
        
        self.model=MotFRCNN(self.im_w, self.im_h)

    def update_config(self):
        cfg[cfg.PHASE].IMS_PER_BATCH=self.batch_size
        cfg.DATA_LOADER_METHOD='INTER_IMG'        
        cfg.STRIDE=self.stride
        cfg.TEMP_MIN_SIZE=64
        cfg.TEMP_MAX_SIZE=min(self.im_h, self.im_w)
        cfg.TEMP_NUM=5
        cfg.GAIN=0.015

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
     
    def vis_anchors(self, image, temp_boxes, det_boxes, search_boxes, track_anchors, fg_anchor_inds):
        '''
        fg_anchors_inds: all inds in a frame, not a box
        '''
        n_instances=len(det_boxes)
        for i in range(n_instances):
            temp_box=temp_boxes[i].astype(np.int32)
            det_box=det_boxes[i].astype(np.int32)
            search_box=search_boxes[i].astype(np.int32)

            cv2.rectangle(image, (temp_box[0],temp_box[1]),(temp_box[2],temp_box[3]),(255,0,0),1)
            cv2.rectangle(image, (det_box[0],det_box[1]),(det_box[2],det_box[3]), (0,0,255), 2)
            cv2.rectangle(image, (search_box[0],search_box[1]),(search_box[2],search_box[3]), (0,255,255), 2)
        
        anchors_all=np.vstack(track_anchors).astype(np.int32)
        for fg_ind in fg_anchor_inds:
            anchor=anchors_all[fg_ind].astype(np.int32)
            ctrx=(anchor[0]+anchor[2])//2
            ctry=(anchor[1]+anchor[3])//2
            cv2.circle(image, (ctrx,ctry), 2, (255,0,0), -1)
            cv2.rectangle(image, (anchor[0],anchor[1]),(anchor[2],anchor[3]), (0,255,0), 1)
            
    def get_param_groups(self, base_lr):
        backbone_lr=base_lr
        if os.path.exists(VGG_PRETRAINED_BN):
            backbone_lr*=self.lr_mult
        return self.model.get_params({'backbone':backbone_lr, 'task':base_lr})
    
    def save_ckpt(self, stepvalues, epoch, iters, lr, logger, log_file):
        model_name='ckpt_mot/dl_mot_iter_{}.pkl'.format(int(iters))
#        model_name='ckpt/dl_mot_iter_{}.pkl'.format(int(iters))
        msg='Snapshotting to {}'.format(model_name)
        logger.info(msg)
        log_file.write(msg+'\n')
        model={'model':self.model.state_dict(),'epoch':epoch, 'iters':iters, 'lr':lr, 'stepvalues':stepvalues, 'im_w':self.im_w, 'im_h':self.im_h}
        torch.save(model, model_name)
    
    def restore_from_ckpt(self, ckpt_path, logger, log_file):
        ckpt=torch.load(ckpt_path)
#        model=ckpt['model']
        epoch=ckpt['epoch']
        iters=ckpt['iters']
        lr=ckpt['lr']
        stepvalues=ckpt['stepvalues']
        msg='Restoring from {}\nStart epoch: {}, total iters: {}, current lr: {}'.format(ckpt_path,\
              epoch, iters, lr)
        logger.info(msg)
        log_file.write(msg+'\n')
#        self.model.update_state_dict(model)
        msg='Model update successfully'
        logger.info(msg)
        log_file.write(msg+'\n')
        return stepvalues, epoch, iters, lr
    
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
        dataset='mot'
        if dataset=='detrac':
            loader=DetracDataLoader(self.im_w, self.im_h, batch_size=self.batch_size)
        else:
            loader=MOTDataLoader(self.im_w, self.im_h, batch_size=self.batch_size)


        logger = logging.getLogger(__name__)

        backbone_pretrained=True
        
        num_samples=loader.get_num_samples()
        
        logger.info('Load {} samples'.format(num_samples))
        
        num_epochs=50
        lr=0.00012
        stepvalues = [30]

        if pretrained_model is None:
            self.model.init_weights()
        else:
            self.model.load_weights(model_path=pretrained_model)
        self.model.cuda()
        
        num_vis_anchors=100
        
        num_jumped=0
        log_file=open('loss.log','w')
        
        iters_per_epoch=int(num_samples/self.batch_size)
        start_epoch=0
        
        if pretrained_model is not None:
            stepvalues, start_epoch, epoch_iters, lr=self.restore_from_ckpt(pretrained_model, logger, log_file)
        else:
            stepvalues, epoch_iters, lr=self.restore_from_epoch(start_epoch, stepvalues, lr, iters_per_epoch)
        
        if backbone_pretrained:
            params=self.get_param_groups(lr)
            optimizer=optim.SGD(params, lr=lr)
            for param_group in optimizer.param_groups:
                print('{} has learning rate {}'.format(param_group['key'], param_group['lr']))
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=lr) 
        logger.info('Start training...')
        
        for epoch in range(start_epoch, num_epochs):
            data_iterator=DataLoader(loader)
            for iters, roidbs in enumerate(data_iterator):
                B=len(roidbs)
                if B==0:
                    msg='No reference image in this minibatch, jump!'
                    logger.info(msg)
                    log_file.write(msg+'\n')
                    num_jumped+=1
                else:
                    '''NHWC'''
                    vis_image_list=[]
                    bbox_overlaps=[]
                    track_anchors=[]
                    for db in roidbs:
                        vis_image_list.append(db['det_image'].squeeze(0).astype(np.uint8, copy=True))
                        if cfg.IMAGE_NORMALIZE:
                            db['temp_image'] -= db['temp_image'].min()
                            db['temp_image'] /= db['temp_image'].max()
                            db['det_image'] -= db['det_image'].min()
                            db['det_image'] /= db['det_image'].max()
                            db['temp_image']=(db['temp_image']-0.5)/0.5
                            db['det_image']=(db['det_image']-0.5)/0.5
                        else:
                            db['temp_image'] -= cfg.PIXEL_MEANS
                            db['det_image'] -= cfg.PIXEL_MEANS
                        bbox_overlaps.append(db['bbox_overlaps'])
                        track_anchors.extend(db['track_anchors'])                                   

                    output_dict=self.model(roidbs)
                    
                    det_anchors=roidbs[0]['anchors']
                    temp_boxes=self.model.temp_boxes
                    det_boxes=self.model.det_boxes
                    search_boxes=self.model.search_boxes
                    temp_classes=self.model.temp_classes
                    det_classes=self.model.det_classes
                    bound=self.model.bound
                    out_size=self.model.out_size
                    num_boxes=self.model.num_boxes
                    
#                    track_anchors=G.gen_region_anchors(self.track_raw_anchors, search_box.reshape(1,-1), bound, K=self.TK, size=(rpn_conv_size, rpn_conv_size))[0]
                    track_anchor_cls_targets, track_anchor_bbox_targets, track_bbox_weights, track_fg_anchor_inds=\
                        T.compute_track_rpn_targets(track_anchors, det_boxes, \
                            bbox_overlaps=bbox_overlaps, bound=bound, rpn_conv_size=self.rpn_conv_size,K=self.TK, num_boxes=num_boxes)
                    
                    gt_boxes, _=U.get_boxes_classes_list(temp_boxes, det_boxes, temp_classes, det_classes, num_boxes)
                    detect_anchor_cls_targets, detect_anchor_bbox_targets, detect_bbox_weights, detect_fg_anchor_inds=\
                        T.compute_detection_rpn_targets(det_anchors, gt_boxes, out_size, \
                             bbox_overlaps=None, K=self.K, batch_size=2*self.batch_size)

                    track_anchor_cls_targets=Variable(torch.from_numpy(track_anchor_cls_targets).long().cuda())
                    track_anchor_bbox_targets=Variable(torch.from_numpy(track_anchor_bbox_targets).float().cuda())                   

                    track_bbox_weights_var=Variable(torch.from_numpy(track_bbox_weights).float().cuda(), requires_grad=False)
                    track_rpn_bbox=torch.mul(output_dict['track_rpn_bbox'], track_bbox_weights_var)
                    track_anchor_bbox_targets=torch.mul(track_anchor_bbox_targets, track_bbox_weights_var)

                    detect_anchor_cls_targets=Variable(torch.from_numpy(detect_anchor_cls_targets).long().cuda())
                    detect_anchor_bbox_targets=Variable(torch.from_numpy(detect_anchor_bbox_targets).float().cuda())                   

                    detect_bbox_weights_var=Variable(torch.from_numpy(detect_bbox_weights).float().cuda(), requires_grad=False)
                    detect_rpn_bbox=torch.mul(output_dict['detect_rpn_bbox'], detect_bbox_weights_var)
                    detect_anchor_bbox_targets=torch.mul(detect_anchor_bbox_targets, detect_bbox_weights_var)

                    if epoch==start_epoch and iters<num_vis_anchors:
                        start=0
                        for i, canvas in enumerate(vis_image_list):
                            left=start
                            right=start+num_boxes[i]
                            canvas_cpy=canvas.copy()
                            self.vis_anchors(canvas_cpy, temp_boxes[left:right], det_boxes[left:right], search_boxes[left:right], track_anchors[left:right], track_fg_anchor_inds[i])
#                            cv2.imwrite('vis_anchors/vis_{}.jpg'.format(iters), canvas_cpy)
                            cv2.imwrite('vis_anchors_mot/vis_{}.jpg'.format(iters), canvas_cpy)
                            start+=num_boxes[i]
                    
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

                    track_rpn_loss_cls=F.cross_entropy(output_dict['track_rpn_logits'], track_anchor_cls_targets, size_average=True, ignore_index=-100)
                    track_rpn_loss_bbox=F.smooth_l1_loss(track_rpn_bbox, track_anchor_bbox_targets, size_average=False, reduce=False)
                    detect_rpn_loss_cls=F.cross_entropy(output_dict['detect_rpn_logits'], detect_anchor_cls_targets, size_average=True, ignore_index=-100)
                    detect_rpn_loss_bbox=F.smooth_l1_loss(detect_rpn_bbox, detect_anchor_bbox_targets, size_average=False, reduce=False)


                    frcnn_loss_cls=F.cross_entropy(output_dict['frcnn_logits'], output_dict['frcnn_cls_target'])
                    frcnn_loss_bbox=F.smooth_l1_loss(output_dict['frcnn_bbox'], output_dict['frcnn_bbox_target'], size_average=False, reduce=False)
                    
                    track_rpn_loss_bbox=torch.div(torch.sum(track_rpn_loss_bbox, dim=1), 4.0)
                    track_rpn_loss_bbox=torch.div(torch.sum(track_rpn_loss_bbox), track_denominator_rpn)
                    detect_rpn_loss_bbox=torch.div(torch.sum(detect_rpn_loss_bbox, dim=1), 4.0)
                    detect_rpn_loss_bbox=torch.div(torch.sum(detect_rpn_loss_bbox), detect_denominator_rpn)
                    
                    frcnn_loss_bbox=torch.div(torch.sum(frcnn_loss_bbox, dim=1), 4.0)
                    frcnn_loss_bbox=torch.div(torch.sum(frcnn_loss_bbox), denominator_frcnn)  
                    
                    '''Do NOT multiply margin in RPN'''
    
                    loss=track_rpn_loss_cls+track_rpn_loss_bbox+detect_rpn_loss_cls+detect_rpn_loss_bbox+frcnn_loss_cls+frcnn_loss_bbox
                    if iters%self.display==0:
                        msg='Epoch {}/{}. Iter_epoch {}/{}. Global_iter: {}. Loss: {}. track_rpn_loss_cls: {}. track_rpn_loss_bbox: {}. detect_rpn_loss_cls: {}. detect_rpn_loss_bbox: {}. frcnn_loss_cls: {}. frcnn_loss_bbox: {}. lr: {}. track_num_examples: {}. detect_num_examples: {}. num_proposals: {}.'.format(epoch, num_epochs, iters, iters_per_epoch, epoch_iters, \
                                        loss.item(), track_rpn_loss_cls.item(), track_rpn_loss_bbox.item(), \
                                        detect_rpn_loss_cls.item(), detect_rpn_loss_bbox.item(), \
                                        frcnn_loss_cls.item(), frcnn_loss_bbox.item(), \
                                        lr, track_num_examples, detect_num_examples, num_fg_proposals)
                        
                        logger.info(msg)
                        log_file.write(msg+'\n')

                    loss_val=loss.cpu().data.numpy()
                        
                    if loss_val > 1e7:
                        msg='Loss too large, stop! {}.'.format(loss_val)
                        logger.error(msg)
                        log_file.write(msg+'\n')
                        assert 0
                    if np.isnan(loss_val):
                        msg='Loss nan, stop! {}.'.format(loss_val)
                        logger.error(msg)
                        log_file.write(msg+'\n')
                        assert 0
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                epoch_iters+=1
                
                if epoch_iters%self.snapshot==0:
                    self.save_ckpt(stepvalues, epoch, epoch_iters, lr, logger, log_file)
                
            if len(stepvalues)>0 and (epoch+1)==stepvalues[0]:
                lr*=self.decay_ratio
                msg='learning rate decay: %e'%lr
                logger.info(msg)
                log_file.write(msg+'\n')
                for param_group in optimizer.param_groups:
                    param_group['lr']=lr
                    if backbone_pretrained and param_group['key']=='backbone':
                        param_group['lr']*=self.lr_mult
                stepvalues.pop(0)
            
        self.save_ckpt(stepvalues, epoch, epoch_iters, lr, logger, log_file)
        
        msg='Finish training!!\nTotal jumped batch: {}'.format(num_jumped)
        logger.info(msg)
        log_file.write(msg+'\n')
        log_file.close()
            
if __name__=='__main__':
    cfg.PHASE='TRAIN'
    cfg.NUM_CLASSES=2
#    pretrained='./dl_mot_pretrained.pkl'
    pretrained='./ckpt/dl_mot_iter_640000.pkl'
    engine=TrainEngine()
#    engine.train(pretrained_model=pretrained)
    engine.train()
