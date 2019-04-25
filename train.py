import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.variable import Variable as Variable

import os
import os.path as op
import numpy as np
import cv2

import rpn.anchor_target as T
from core.config import cfg
from time import time

DEBUG=False

def draw_track_anchors(image, temp_boxes, det_boxes, search_boxes, track_anchors, fg_anchor_inds):
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

def draw_detect_anchors(image, boxes, anchors, fg_inds):
    for box in boxes:
        box=box.astype(np.int32)
        cv2.rectangle(image, (box[0],box[1]),(box[2],box[3]),(0,0,255),2)
    fg_anchors=anchors[fg_inds]
    for box in fg_anchors:
        box=box.astype(np.int32)
        cv2.rectangle(image, (box[0],box[1]),(box[2],box[3]),(0,255,0),1)

'''
params keys:
epoch
start_epoch
num_epochs
epoch_iters
lr
out_size
K
vis_anchors_dir
display
num_vis_anchors
'''
def train_epoch(model, data_loader, optimizer, logger, params):
    for iters, roidbs in enumerate(data_loader):
        B=len(roidbs)
        if B==0:
            #msg='No reference image in this minibatch, jump!'
            #logger.info(msg)
            pass
        else:
            '''NHWC'''
            overlaps=[]
            track_anchors=[]
            for db in roidbs:
                if cfg.IMAGE_NORMALIZE:
                    db['temp_image'] /= 255.0
                    db['det_image'] /= 255.0
                else:
                    db['temp_image'] -= cfg.PIXEL_MEANS
                    db['det_image'] -= cfg.PIXEL_MEANS
                overlaps.append(db['bbox_overlaps'])
                track_anchors.extend(db['track_anchors'])                                   
            tic=time()
            output_dict=model(roidbs)
            toc=time()
            if DEBUG:
                print('Forward costs {}s'.format(toc-tic))
            det_anchors=roidbs[0]['anchors']
            temp_boxes=output_dict['temp_boxes']
            det_boxes=output_dict['det_boxes']
            search_boxes=output_dict['search_boxes']
            num_boxes=output_dict['num_boxes']

            tic=time()
            track_anchor_cls_targets, track_anchor_bbox_targets, track_bbox_weights, track_fg_anchor_inds=\
                T.compute_track_rpn_targets(track_anchors, det_boxes, \
                    bbox_overlaps=overlaps, bound=params['bound'], rpn_conv_size=params['rpn_conv_size'], K=params['TK'], num_boxes=num_boxes)
            
            gt_boxes=output_dict['gt_boxes']
            detect_anchor_cls_targets, detect_anchor_bbox_targets, detect_bbox_weights, detect_fg_anchor_inds=\
                T.compute_detection_rpn_targets(det_anchors, gt_boxes, params['out_size'], \
                        bbox_overlaps=None, K=params['K'], batch_size=output_dict['B'])

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

            if params['epoch']==params['start_epoch'] and iters<params['num_vis_anchors']:
                start=0
                d1=params['vis_anchors_dirs'][0]
                d2=params['vis_anchors_dirs'][1]
                if not op.exists(d1):
                    os.mkdir(d1)
                if not op.exists(d2):
                    os.mkdir(d2)
                gt_boxes=output_dict['gt_boxes']
                for i, db in enumerate(roidbs):
                    left=start
                    right=start+num_boxes[i]
                    canvas_track=(db['det_image'].squeeze(0) + cfg.PIXEL_MEANS).astype(np.uint8, copy=True)
                    canvas_det=(db['temp_image'].squeeze(0) + cfg.PIXEL_MEANS).astype(np.uint8, copy=True)
                    draw_track_anchors(canvas_track, temp_boxes[left:right], det_boxes[left:right], search_boxes[left:right], track_anchors[left:right], track_fg_anchor_inds[i])
                #                            cv2.imwrite('vis_anchors/vis_{}.jpg'.format(iters), canvas_cpy)
                    draw_detect_anchors(canvas_det, gt_boxes[i], det_anchors, detect_fg_anchor_inds[i])
                    cv2.imwrite('{}/vis_{}.jpg'.format(d1,iters), canvas_track)
                    cv2.imwrite('{}/vis_{}.jpg'.format(d2,iters), canvas_det)
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
            if iters%params['display']==0:
                msg='Epoch {}/{}. Iter_epoch {}/{}. Loss: {}. track_rpn_loss_cls: {}. track_rpn_loss_bbox: {}. detect_rpn_loss_cls: {}. detect_rpn_loss_bbox: {}. frcnn_loss_cls: {}. frcnn_loss_bbox: {}. lr: {}. track_num_examples: {}. detect_num_examples: {}. num_proposals: {}.'.format(params['epoch'], params['num_epochs'], iters, params['epoch_iters'], \
                                loss.item(), track_rpn_loss_cls.item(), track_rpn_loss_bbox.item(), \
                                detect_rpn_loss_cls.item(), detect_rpn_loss_bbox.item(), \
                                frcnn_loss_cls.item(), frcnn_loss_bbox.item(), \
                                params['lr'], track_num_examples, detect_num_examples, num_fg_proposals)
                
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
