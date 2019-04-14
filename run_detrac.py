import os.path as op
import numpy as np
import cv2
from model.mot_forward import nms_cuda, MotFRCNN
from core.config import cfg
from dataset import detrac,vid,vot,kitti,video,mot
from dataset.data_loader import DataLoader
from inference import get_detect_output, get_track_output
import roidb.box_utils as butil
import rpn.generate_anchors as G
from rpn.util import bbox_transform_inv, bbox_overlap_frcnn
from rpn.template import get_template

im_width=640
im_height=384

CLASSES=['__background', 'car', 'van', 'bus', 'truck']
cfg.PHASE='TEST'
cfg.NUM_CLASSES=len(CLASSES)
cfg.DET_SCORE_THRESH=0.95
cfg.TRACK_SCORE_THRESH=0.0
cfg.TRACK_MAX_DIST=20
cfg.TEST.RPN_NMS_THRESH=0.6
cfg.TEMP_MIN_SIZE=64
cfg.TEMP_MAX_SIZE=im_height
cfg.TEMP_NUM=4

MAX_TEMPLATE_SIZE=im_height

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

templates=get_template(min_size=cfg.TEMP_MIN_SIZE, max_size=cfg.TEMP_MAX_SIZE, num_templates=cfg.TEMP_NUM)
det_raw_anchors=G.generate_anchors(cfg.BASIC_SIZE, cfg.RATIOS, cfg.SCALES)
track_raw_anchors=G.generate_anchors(cfg.TRACK_BASIC_SIZE, cfg.TRACK_RATIOS, cfg.TRACK_SCALES)
K=len(cfg.RATIOS)*len(cfg.SCALES)
TK=len(cfg.TRACK_RATIOS)*len(cfg.TRACK_SCALES)
rpn_conv_size=cfg.RPN_CONV_SIZE
bound=(im_width, im_height)
out_size=(im_width//8, im_height//8)

dummy_search_box=np.array([[0,0,im_width-1,im_height-1]])
det_anchors=G.gen_region_anchors(det_raw_anchors, dummy_search_box, bound, K=K, size=out_size)[0]

configs={}
configs['K']=TK
configs['search_boxes']=None
configs['rpn_conv_size']=rpn_conv_size
configs['raw_anchors']=track_raw_anchors
configs['bound']=bound

class Target(object):
    def __init__(self, bbox, obj_id):
        self.bbox=bbox
        self.obj_id=obj_id
        self.obj_id2=obj_id

def bbox_overlaps(bbox, bboxes):
    w,h=bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1
    ws,hs=bboxes[:,2]-bboxes[:,0]+1, bboxes[:,3]-bboxes[:,1]+1
    area=w*h
    areas=ws*hs

    x1=np.maximum(bbox[0], bboxes[:,0])
    y1=np.maximum(bbox[1], bboxes[:,1])
    x2=np.minimum(bbox[2], bboxes[:,2])
    y2=np.minimum(bbox[3], bboxes[:,3])

    intersec=np.maximum(0, (x2-x1+1)*(y2-y1+1))
    ious=1.0*intersec/(areas+area)

    return ious

def check_boxes(boxes):
    x1,y1,x2,y2=np.split(boxes, 4, axis=1)
    ws,hs=x2-x1+1,y2-y1+1
    good_inds=np.where(np.bitwise_and(ws<MAX_TEMPLATE_SIZE-1, hs<MAX_TEMPLATE_SIZE-1)==1)[0]
    return good_inds

def detect_track(model, roidb=None):
    output_dict=model(roidb, task='all')

    configs['search_boxes']=roidb['search_boxes']
    configs['temp_boxes']=roidb['temp_boxes']

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
        bboxes=bbox_with_score[:,:4]
        ws,hs=bboxes[:,2]-bboxes[:,0]+1,bboxes[:,3]-bboxes[:,1]+1
        valid_inds=np.where(np.bitwise_and(ws<MAX_TEMPLATE_SIZE,hs<MAX_TEMPLATE_SIZE)==1)[0]
        temp_boxes=np.append(temp_boxes, bboxes[valid_inds], 0) 
    return temp_boxes

def main(dataset_obj, model):
    targets=[]

    loader=DataLoader(dataset_obj)

    temp_boxes=None
    det_boxes=None
    track_started=False
    num_instances=0

    det_interval=30
    cur_max_obj_id=0

    roidb={}
    roidb['bound']=bound
    roidb['temp_classes']=None
    roidb['det_classes']=None
    roidb['temp_boxes']=None
    roidb['det_boxes']=None
    roidb['anchors']=det_anchors

#    writer=cv2.VideoWriter('./self_video.avi', cv2.VideoWriter_fourcc('M','J','P','G'),30.0,(im_width, im_height))
    for idx, image in enumerate(loader):
        canvas=image.copy()
        canvas_det=image.copy()
        if not track_started or temp_boxes.shape[0]==0:
            ret=detect(model, image, roidb)
            cls_bboxes=ret['cls_bboxes']
            det_boxes=cls_bboxes_to_boxes(cls_bboxes)
            temp_boxes=det_boxes
            '''init temp_boxes'''
            roidb['temp_boxes']=temp_boxes                        
            '''init num_instances'''
            num_instances=temp_boxes.shape[0]
            for i in range(num_instances):
                t=Target(temp_boxes[i], i)
                targets.append(t)
            cur_max_obj_id=num_instances
            track_started=True
        else:
            temp_boxes=[]
            search_boxes=[]
            target_ids=[]
            for ix, t in enumerate(targets):
                if t.obj_id>=0:
                    temp_boxes.append(t.bbox)
                    _,best_template=butil.best_search_box_test(templates, t.bbox, bound)
                    search_boxes.append(best_template)
                    
                    target_ids.append(ix)

            temp_boxes=np.array(temp_boxes)
            search_boxes=np.array(search_boxes)

            temp_image=roidb['det_image']
            det_image=image.copy()
            roidb['temp_image']=temp_image
            roidb['det_image']=det_image
            roidb['good_inds']=check_boxes(temp_boxes)
            
            roidb['search_boxes']=search_boxes
            
            det_ret, track_ret=detect_track(model, roidb)

            bboxes_list=track_ret['bboxes_list']
            
            all_obj_tracked=True
            for ix, bboxes in enumerate(bboxes_list):
                 if len(bboxes)==0:
                     all_obj_tracked=False
                     targets[target_ids[ix]].obj_id=-1
                 else:
                     targets[target_ids[ix]].bbox=bboxes.squeeze(0)[:4]

            if not all_obj_tracked or idx%det_interval==0:
                cls_bboxes=det_ret['cls_bboxes']
                det_boxes=cls_bboxes_to_boxes(cls_bboxes)
                new_bboxes=[]
                for ix, bboxes in enumerate(det_boxes):
                    det_box=bboxes.squeeze(0)[:4]
                    overlaps=bbox_overlaps(det_box, temp_boxes)
                    max_overlaps=np.max(overlaps)
                    if max_overlaps>0.7:
                        ind=np.argmax(overlaps)
                        if targets[target_ids[ind]].obj_id==-1:
                            targets[target_ids[ind]].bbox=det_box
                            targets[target_ids[ind]].obj_id=targets[target_ids[ind]].obj_id2    #recover
                    else:
                        new_bboxes.append(det_box)

                for det_box in new_bboxes:
                    t=Target(cur_max_obj_id, det_box)
                    targets.append(t)
                    cur_max_obj_id+=1

        for t in targets:
            obj_id=t.obj_id
            if obj_id>=0:
                bbox=t.bbox.astype(np.int32)
                cv2.rectangle(canvas, (bbox[0],bbox[1]),(bbox[2],bbox[3]), colors[obj_id%len(colors)], 2)
        
        for ix, det_box in enumerate(det_boxes):
            det_box=det_box.astype(np.int32)
            cv2.rectangle(canvas_det, (det_box[0],det_box[1]),(det_box[2],det_box[3]), colors[ix%len(colors)], 2)
        
        print('Frame {}'.format(idx))
        cv2.imshow('track', canvas)
        cv2.imshow('det',canvas_det)
#        writer.write(canvas)
        if cv2.waitKey(1)==27:
            break
            

if __name__=='__main__':
    model_path='./ckpt/dl_mot_epoch_14.pkl'
    dataset_obj=detrac.Detrac(im_width=im_width, im_height=im_height, name='DETRAC', load_gt=False)
    dataset_obj.choice('MVI_40991')

    model=MotFRCNN(im_width, im_height, pretrained=False)
    model.load_weights(model_path)
    model.cuda()
    
    main(dataset_obj, model=model)