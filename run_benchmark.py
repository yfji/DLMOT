import os.path as op
import numpy as np
import cv2
from model.mot_forward import nms_cuda, MotFRCNN
from core.config import cfg
from dataset import Detrac,KITTI,Video
from dataset.data_loader import DataLoader
from inference import CLASSES, get_detect_output, get_track_output
import roidb.box_utils as butil
import rpn.generate_anchors as G
from rpn.util import bbox_transform_inv, bbox_overlap_frcnn
from rpn.template import get_template

im_width=640
im_height=384

MAX_TEMPLATE_SIZE=300

colors = [ [0, 255, 255], [255, 85, 0], 
              [255, 170, 0], [255, 255, 0], 
              [170, 255, 0], [85, 255, 0], 
              [0, 255, 0],   [0, 255, 85], 
              [0, 255, 170], [255, 0, 0],
              [0, 170, 255], [0, 85, 255], 
              [0, 0, 255], [85, 0, 255],
              [255, 0, 0], [255, 85, 0], 
              [255, 170, 0], [255, 255, 0],
              [0, 0, 255], [85, 0, 255],
              [255, 0, 0], [255, 85, 0], 
              [255, 170, 0], [255, 255, 0]]

def load_parameters(model, model_path=None):
    model.load_weights(model_path=model_path)
    params={}
    params['templates']=get_template(min_size=cfg.TEMP_MIN_SIZE, max_size=cfg.TEMP_MAX_SIZE, num_templates=cfg.TEMP_NUM)
    print('Using templates:')
    print(params['templates'])
    params['K']=len(cfg.RATIOS)*len(cfg.SCALES)
    params['TK']=len(cfg.TRACK_RATIOS)*len(cfg.TRACK_SCALES)
    params['rpn_conv_size']=cfg.RPN_CONV_SIZE
    params['bound']=(im_width, im_height)
    params['out_size']=(im_width//8, im_height//8)

    track_raw_anchors=G.generate_anchors(cfg.TRACK_BASIC_SIZE, cfg.TRACK_RATIOS, cfg.TRACK_SCALES)
    params['track_raw_anchors']=track_raw_anchors
    dummy_search_box=np.array([[0,0,im_width-1,im_height-1]])
    det_raw_anchors=G.generate_anchors(cfg.BASIC_SIZE, cfg.RATIOS, cfg.SCALES)

    anchors=G.gen_region_anchors(det_raw_anchors, dummy_search_box, params['bound'], K=params['K'], size=params['out_size'])[0]
    params['anchors']=anchors
    
    configs={}
    configs['K']=params['TK']
    configs['search_boxes']=None
    configs['rpn_conv_size']=params['rpn_conv_size']
    configs['raw_anchors']=params['track_raw_anchors']
    configs['bound']=params['bound']

    return params, configs

def add_new_targets(ids, boxes, new_boxes):
    combined_ids=ids.copy()
    combined_boxes=boxes.copy()
    num_instances=len(ids)
    num_new_instances=new_boxes.shape[0]
    
    index=0
    new_index=0
    for i in range(num_instances):
        if combined_ids[i]==0 and new_index<num_new_instances:
            new_box=new_boxes[new_index]
            w,h=new_box[2]-new_box[0]+1,new_box[3]-new_box[1]+1
            if w<MAX_TEMPLATE_SIZE and h<MAX_TEMPLATE_SIZE:
                 combined_ids[i]=1    #the ith target is lost and replaced with a new target in its position
                 combined_boxes=np.insert(combined_boxes, i, new_boxes[new_index], axis=0)
                 new_index+=1
            index+=1
        index+=1
    if new_index<num_new_instances:
        combined_ids=np.append(combined_ids, np.ones(num_new_instances-new_index))
        combined_boxes=np.append(combined_boxes, new_boxes[new_index:,:], 0)
    combined_ids=combined_ids[:np.max(np.where(combined_ids==1)[0])]
    return combined_ids, combined_boxes

def check_boxes(boxes):
    x1,y1,x2,y2=np.split(boxes, 4, axis=1)
    ws,hs=x2-x1+1,y2-y1+1
    good_inds=np.where(np.bitwise_and(ws<MAX_TEMPLATE_SIZE-1, hs<MAX_TEMPLATE_SIZE-1)==1)[0]
    return good_inds

def sort_boxes(track_boxes, det_boxes):
    bbox_overlaps=bbox_overlap_frcnn(det_boxes, track_boxes)
    max_overlaps=np.max(bbox_overlaps, axis=1)
    new_target_inds=np.where(max_overlaps<0.2)[0]

    temp_boxes=np.append(track_boxes, det_boxes[new_target_inds], 0)
    return temp_boxes

def detect_track(model, temp_image, det_image, temp_boxes, roidb=None, params=None, configs=None):
    if roidb is None:
        roidb={}
    roidb['bound']=params['bound']
    roidb['temp_image']=temp_image[np.newaxis,:,:,:].astype(np.float32)
    roidb['det_image']=det_image[np.newaxis,:,:,:].astype(np.float32)
    roidb['good_inds']=check_boxes(temp_boxes)
    roidb['temp_boxes']=temp_boxes
    search_boxes=[]
    for temp_box in temp_boxes:
        _,best_template=butil.best_search_box_test(params['templates'], temp_box, params['bound'])
        search_boxes.append(best_template)
    
    search_boxes=np.array(search_boxes)
    roidb['det_boxes']=None
    roidb['det_classes']=None
    roidb['temp_classes']=None
    roidb['search_boxes']=search_boxes
    roidb['anchors']=params['anchors']

    output_dict=model(roidb, task='all')
    configs['search_boxes']=search_boxes
    configs['temp_boxes']=temp_boxes

    det_rets=get_detect_output(output_dict, batch_size=1)
    track_ret=get_track_output(output_dict, configs)
    det_ret=det_rets[0]
    '''dicts'''
    return det_ret, track_ret
    
def detect(model, image, roidb=None):
    if roidb is None:
        roidb={}
    roidb['temp_image']=image[np.newaxis,:,:,:].astype(np.float32, copy=True)
    roidb['det_image']=image[np.newaxis,:,:,:].astype(np.float32, copy=True)
    output_dict=model(roidb, task='det')
    rets=get_detect_output(output_dict, batch_size=1)    
    return rets[0]

def cls_bboxes_to_boxes(cls_bboxes):
    temp_boxes=np.zeros((0,4),dtype=np.float32)
    for i in range(1, len(CLASSES)):
        bbox_with_score=cls_bboxes[i]
        if len(bbox_with_score)==0:
            continue                
#        scores=bbox_with_score[:,-1]
#        fg_inds=np.where(scores>0.8)
#        bbox_with_score=bbox_with_score[fg_inds]
        bboxes=bbox_with_score[:,:4]
        ws,hs=bboxes[:,2]-bboxes[:,0]+1,bboxes[:,3]-bboxes[:,1]+1
        valid_inds=np.where(np.bitwise_and(ws<MAX_TEMPLATE_SIZE,hs<MAX_TEMPLATE_SIZE)==1)[0]
        temp_boxes=np.append(temp_boxes, bboxes[valid_inds], 0) 
    return temp_boxes

def main(dataset_obj, model, params, configs):
    loader=DataLoader(dataset_obj)

    temp_boxes=None
    det_boxes=None
    track_started=False
    num_instances=0
    track_ids=None

    frame_ind=0
    det_interval=30

    roidb={}
    roidb['bound']=params['bound']
    roidb['anchors']=params['anchors']

#    writer=cv2.VideoWriter('./self_video.avi', cv2.VideoWriter_fourcc('M','J','P','G'),30.0,(im_width, im_height))
    for idx, image in enumerate(loader):
        canvas=image.copy()
        canvas_det=image.copy()
        if not track_started or temp_boxes.shape[0]==0:
            ret=detect(model, image, roidb)
            cls_bboxes=ret['cls_bboxes']
            temp_boxes=cls_bboxes_to_boxes(cls_bboxes)
            '''init temp_boxes'''
            roidb['temp_boxes']=temp_boxes                        
            '''init num_instances'''
            num_instances=temp_boxes.shape[0]
            track_ids=np.ones(num_instances, dtype=np.int32)
            track_started=True
        else:
            temp_boxes=roidb['temp_boxes']
#            last_temp_boxes=temp_boxes.copy()

            temp_image=roidb['det_image'].squeeze(0)
            det_image=image.copy()
            roidb['temp_image']=temp_image
            roidb['det_image']=det_image
            roidb['good_inds']=check_boxes(temp_boxes)
            
            search_boxes=[]
            for temp_box in temp_boxes:
                _,best_template=butil.best_search_box_test(params['templates'], temp_box, params['bound'])
                search_boxes.append(best_template)
            
            search_boxes=np.array(search_boxes)
            roidb['search_boxes']=search_boxes
            
            det_ret, track_ret=detect_track(model, temp_image, det_image, temp_boxes, roidb, params=params, configs=configs)
            '''do something with det_ret'''
            bboxes_list=track_ret['bboxes_list']
            valid_boxes=np.zeros((0,4), dtype=np.float32)
                
            index=0
            for i in range(num_instances):
                if not track_ids[i]:
                    continue
                bboxes=np.array(bboxes_list[index])
                if len(bboxes.shape)>1:
                    bboxes=bboxes.squeeze()
                if bboxes.size==0:
                    track_ids[i]=0
                else:
                    bbox=bboxes[:4]
                    w,h=bbox[2]-bbox[0]+1,bbox[3]-bbox[1]+1
                    if w>=MAX_TEMPLATE_SIZE or h>=MAX_TEMPLATE_SIZE:
                         track_ids[i]=0
                    else:
                         valid_boxes=np.append(valid_boxes, bbox.reshape(1,-1), 0)
                index+=1                 
            if valid_boxes.shape[0]!=temp_boxes.shape[0] or frame_ind==det_interval:
                if frame_ind!=det_interval:
                    print('Target changed, use det_ret!')    
                else:
                    print('Det interval arrived, use det_ret!')
                    frame_ind=0      
                cls_bboxes=det_ret['cls_bboxes']
                det_boxes=cls_bboxes_to_boxes(cls_bboxes)
                if valid_boxes.shape[0]>0 and det_boxes.shape[0]>0:
                    det_boxes=sort_boxes(valid_boxes, det_boxes)
#                track_ids, temp_boxes=add_new_targets(track_ids, valid_boxes, det_boxes[num_instances:])                    
                track_ids, temp_boxes=add_new_targets(track_ids, valid_boxes, det_boxes[valid_boxes.shape[0]:])
                num_instances=len(track_ids)
            else:
                temp_boxes=valid_boxes.copy()
            
            roidb['temp_boxes']=temp_boxes
        '''visualize'''
        index=0
#        print(track_ids)
#        print(temp_boxes.shape[0])
        for i in range(num_instances):
            if not track_ids[i]:
                continue
            bbox=temp_boxes[index].astype(np.int32)
            cv2.rectangle(canvas, (bbox[0], bbox[1]), (bbox[2],bbox[3]), tuple(colors[i%len(colors)]), 2)                
            index+=1
        if det_boxes is not None:
            for bbox in det_boxes:
                bbox=bbox.astype(np.int32)
                cv2.rectangle(canvas_det, (bbox[0], bbox[1]), (bbox[2],bbox[3]), tuple(colors[i%len(colors)]), 2)                
        frame_ind+=1
        print('Frame {}'.format(idx))
        cv2.imshow('track', canvas)
        cv2.imshow('det',canvas_det)
#        writer.write(canvas)
        if cv2.waitKey(1)==27:
            break

if __name__=='__main__':
    cfg.PHASE='TEST'
    cfg.NUM_CLASSES=5
    cfg.DET_SCORE_THRESH=0.95
    cfg.TRACK_SCORE_THRESH=0.0
    cfg.TRACK_MAX_DIST=20
    cfg.TEST.RPN_NMS_THRESH=0.6

    dataset='detrac'
    dataset_obj=None
    model_path=None
    if dataset=='detrac':
        model_path='./ckpt/dl_mot_epoch_6.pkl'
        dataset_obj=Detrac(im_width=im_width, im_height=im_height, name='DETRAC', load_gt=False)
        dataset_obj.choice('MVI_39761')
#        dataset_obj.choice('MVI_40201')        
    elif dataset=='kitti':
        model_path='./ckpt/dl_mot_iter_800000.pkl'
        dataset_obj=KITTI(im_width=im_width, im_height=im_height, name='KITTI')
        dataset_obj.choice('0000')

    elif dataset=='video':
        dataset_obj=Video(im_width=im_width, im_height=im_height, name='VIDEO')
        dataset_obj.choice('beisanhuan.mp4')
    
    model=MotFRCNN(im_width, im_height, pretrained=False)
    params, configs=load_parameters(model, model_path=model_path)
    model.cuda()
    
    main(dataset_obj, model, params, configs)