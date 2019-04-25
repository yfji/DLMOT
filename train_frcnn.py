import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.variable import Variable as Variable

import os
import numpy as np
import cv2

from model.mot_frcnn import MotFRCNN
from roidb import DataLoader, DetracDataReader
import rpn.generate_anchors as G
import rpn.anchor_target as T
import rpn.util as U
from core.config import cfg
from train import *
from mynn import DataParallel

import logging

logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(levelname)s - %(message)s')

BBOX_OVERLAP_PRECOMPUTED=False
RESOLUTION={'VGG':(640,384),'RESNET':(640,384),'ALEX':(399,255)}
STRIDE={'VGG':8,'RESNET':8,'ALEX':8}
net_type='RESNET'

num_gpus=1
gpu_ids=0

def set_gpu_id():
    global num_gpus, gpu_ids
    gpu_id_list=cfg.GPU_ID.split(',')
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.GPU_ID
    print('Using GPUs {}'.format(cfg.GPU_ID))
    gpu_ids=list(range(len(gpu_id_list)))

class TrainEngine(object):
    def __init__(self):
        self.batch_size=2
        
        self.stride=STRIDE[net_type]   
        self.im_w, self.im_h=RESOLUTION[net_type]

        self.backbone_pretrained=True        
        self.lr_mult=0.5
        self.decay_ratio=0.1
        self.snapshot=2
        
        self.fetch_config()
        self.update_config()
        
        self.K=len(self.ratios)*len(self.scales)
        self.TK=len(self.track_ratios)*len(self.track_scales)                    

        self.raw_anchors=G.generate_anchors(self.basic_size, self.ratios, self.scales)
        self.track_raw_anchors=G.generate_anchors(self.track_basic_size, self.track_ratios, self.track_scales)
        
        self.model=MotFRCNN(self.im_w, self.im_h)

    def update_config(self):
        cfg[cfg.PHASE].IMS_PER_BATCH=self.batch_size
        cfg.STRIDE=self.stride
        cfg.TEMP_MIN_SIZE=48
        cfg.TEMP_MAX_SIZE=300
        cfg.TEMP_NUM=7
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
    
            
    def get_param_groups(self, base_lr):
        backbone_lr=base_lr
        if self.backbone_pretrained:
            backbone_lr*=self.lr_mult
        return self.model.get_params({'backbone':backbone_lr, 'task':base_lr})
    
    def save_ckpt(self, stepvalues, epoch, iters, lr, logger):
        model_name='ckpt/dl_mot_epoch_{}.pkl'.format(epoch)
        msg='Snapshotting to {}'.format(model_name)
        logger.info(msg)
        model={'model':self.model.state_dict(),'epoch':epoch, \
            'iters':iters, 'lr':lr, 'stepvalues':stepvalues, \
            'temp_min_size':cfg.TEMP_MIN_SIZE,'temp_max_size':cfg.TEMP_MAX_SIZE,\
            'temp_num':cfg.TEMP_NUM,'im_w':self.im_w, 'im_h':self.im_h}
        torch.save(model, model_name)
    
    def restore_from_ckpt(self, ckpt_path, logger):
        ckpt=torch.load(ckpt_path)
#        model=ckpt['model']
        epoch=ckpt['epoch']
        iters=ckpt['iters']
        lr=ckpt['lr']
        cfg.TEMP_MIN_SIZE=ckpt['temp_min_size']
        cfg.TEMP_MAX_SIZE=ckpt['temp_max_size']
        cfg.TEMP_NUM=ckpt['temp_num']
        stepvalues=ckpt['stepvalues']
        msg='Restoring from {}\nStart epoch: {}, total iters: {}, current lr: {}'.format(ckpt_path,\
              epoch, iters, lr)
        logger.info(msg)
        logger.info('Model update successfully')
        return stepvalues, epoch, iters, lr
    
    def train(self, pretrained_model=None):
        detrac_reader=DetracDataReader(self.im_w, self.im_h, batch_size=self.batch_size)
        data_loader=DataLoader(detrac_reader, shuffle=True, batch_size=self.batch_size, num_workers=2)

        logger = logging.getLogger(__name__)

        num_samples=detrac_reader.__len__()
        num_samples=detrac_reader.__len__()
        epoch_iters=num_samples//self.batch_size

        num_epochs=50
        lr=0.0002
        stepvalues = [30]        

        num_vis_anchors=100
        config_params={'lr':lr,'epoch':0,'start_epoch':0,'num_epochs':num_epochs,'epoch_iters':epoch_iters,\
            'out_size':detrac_reader.out_size,'bound':detrac_reader.bound,'K':self.K,'TK':self.TK,\
            'rpn_conv_size':self.rpn_conv_size,
            'vis_anchors_dirs':['./vis_anchors/track','./vis_anchors/det'],'display':20,
            'num_vis_anchors':num_vis_anchors}
        
        logger.info('Load {} samples'.format(num_samples))
        
        if pretrained_model is None:
            self.model.init_weights()
        else:
            self.model.load_weights(model_path=pretrained_model)
        self.model.cuda()
        
        num_vis_anchors=100
        
        num_jumped=0
        
        start_epoch=0
        
        if pretrained_model is not None:
            stepvalues, start_epoch, epoch_iters, lr=self.restore_from_ckpt(pretrained_model, logger)
        
        if self.backbone_pretrained:
            params=self.get_param_groups(lr)
            optimizer=optim.SGD(params, lr=lr)
            for param_group in optimizer.param_groups:
                print('{} has learning rate {}'.format(param_group['key'], param_group['lr']))
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=lr) 

        if num_gpus>1:
            self.model=DataParallel(self.model, device_ids=gpu_ids)

        logger.info('Start training...')
        
        for epoch in range(start_epoch, num_epochs):
            config_params['epoch']=epoch
            train_epoch(self.model, data_loader, optimizer, logger, config_params)
                
            if epoch>0 and epoch%self.snapshot==0:
                self.save_ckpt(stepvalues, epoch, epoch_iters, lr, logger)
            if len(stepvalues)>0 and (epoch+1)==stepvalues[0]:
                lr*=self.decay_ratio
                msg='learning rate decay: %e'%lr
                logger.info(msg)
                for param_group in optimizer.param_groups:
                    param_group['lr']=lr
                    if self.backbone_pretrained and param_group['key']=='backbone':
                        param_group['lr']*=self.lr_mult
                stepvalues.pop(0)
            
        self.save_ckpt(stepvalues, epoch, epoch_iters, lr, logger)
        
        msg='Finish training!!\nTotal jumped batch: {}'.format(num_jumped)
        logger.info(msg)
            
if __name__=='__main__':
    set_gpu_id()
    cfg.PHASE='TRAIN'
#    pretrained='./dl_mot_pretrained.pkl'
    pretrained='./ckpt/dl_mot_iter_640000.pkl'
    engine=TrainEngine()
#    engine.train(pretrained_model=pretrained)
    engine.train()
