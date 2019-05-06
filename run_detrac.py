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
from rpn.util import bbox_transform_inv
from rpn.template import get_template

im_width=640
im_height=384

CLASSES=['__background', 'car', 'van', 'bus', 'truck']


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
    global MAX_TEMPLATE_SIZE
    MAX_TEMPLATE_SIZE = cfg.TEMP_MAX_SIZE
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

class Target(object):
    def __init__(self, obj_id, bbox):
        self.bbox=bbox
        self.obj_id=obj_id
        self.obj_id2=obj_id

        self.history=[]

def box_overlaps(bbox, bboxes):
    w,h=bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1
    ws,hs=bboxes[:,2]-bboxes[:,0]+1, bboxes[:,3]-bboxes[:,1]+1
    area=w*h
    areas=ws*hs

    x1=np.maximum(bbox[0], bboxes[:,0])
    y1=np.maximum(bbox[1], bboxes[:,1])
    x2=np.minimum(bbox[2], bboxes[:,2])
    y2=np.minimum(bbox[3], bboxes[:,3])

    iws=np.maximum(0,x2-x1+1)
    ihs=np.maximum(0,y2-y1+1)

    intersec=np.maximum(0, iws*ihs)
    ious=1.0*intersec/(areas+area-intersec)

    return ious

def check_boxes(boxes):
    x1,y1,x2,y2=np.split(boxes, 4, axis=1)
    ws,hs=x2-x1+1,y2-y1+1
    sz=np.maximum(ws,hs)
    good_inds=np.where(sz<MAX_TEMPLATE_SIZE-1)[0]
    return good_inds

def detect_track(model, roidb=None, configs=None):
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

def main(dataset_obj, model, params=None, configs=None):
    targets=[]

    loader=DataLoader(dataset_obj)

    temp_boxes=None
    det_boxes=None
    track_started=False
    num_instances=0

    det_interval=30
    cur_max_obj_id=0

    roidb={}
    roidb['bound']=params['bound']
    roidb['anchors']=params['anchors']

    template_anchors=G.gen_template_anchors(params['track_raw_anchors'], params['templates'], params['TK'], size=(params['rpn_conv_size'],params['rpn_conv_size']))
    # writer=cv2.VideoWriter('./self_video.avi', cv2.VideoWriter_fourcc('M','J','P','G'),20.0,(im_width, im_height))
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
                t=Target(i,temp_boxes[i])
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
                    best_template, anchors=butil.best_search_box_test(params['templates'], t.bbox, template_anchors, params['bound'])
                    search_boxes.append(best_template)
                    
                    target_ids.append(ix)

            temp_boxes=np.array(temp_boxes)
            search_boxes=np.array(search_boxes)
            
            good_inds=check_boxes(temp_boxes)
            temp_boxes=temp_boxes[good_inds]
            search_boxes=search_boxes[good_inds]
            
            assert temp_boxes.shape[0]==search_boxes.shape[0], 'Error in dim0, run_detrac'

            temp_image=roidb['det_image']
            det_image=image[np.newaxis,:,:,:].astype(np.float32)
            roidb['temp_image']=temp_image
            roidb['det_image']=det_image
            roidb['good_inds']=good_inds
            
            roidb['search_boxes']=search_boxes
            roidb['temp_boxes']=temp_boxes
            
            det_ret, track_ret=detect_track(model, roidb, configs)

            bboxes_list=track_ret['bboxes_list']
            
            all_obj_tracked=True
            for ix, bboxes in enumerate(bboxes_list):
                 if len(bboxes)==0:
                     all_obj_tracked=False
                     targets[target_ids[ix]].obj_id=-1
                 else:
                     targets[target_ids[ix]].bbox=bboxes.ravel()[:4]

            if not all_obj_tracked or idx%det_interval==0:
                cls_bboxes=det_ret['cls_bboxes']
                det_boxes=cls_bboxes_to_boxes(cls_bboxes)
                new_bboxes=[]
                comp_boxes=[]
                for i in range(temp_boxes.shape[0]):
                    if len(bboxes_list[i])>0:
                        comp_boxes.append(bboxes_list[i].ravel()[:4])
                    else:
                        comp_boxes.append(temp_boxes[i])
                comp_boxes=np.array(comp_boxes)
                for ix, bboxes in enumerate(det_boxes):
                    print(bboxes)
                    det_box=bboxes[:4]
                    # overlaps=box_overlaps(det_box, temp_boxes)
                    overlaps=box_overlaps(det_box, comp_boxes)
                    max_overlap=np.max(overlaps)
                    # print('max overlap: {}. temp boxes number: {}'.format(max_overlap, comp_boxes.shape[0]))
                    if max_overlap>0.5:
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
                cv2.putText(canvas, 'obj_{}'.format(obj_id), (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, colors[obj_id%len(colors)], 1)
        
        for ix, det_box in enumerate(det_boxes):
            det_box=det_box.astype(np.int32)
            cv2.rectangle(canvas_det, (det_box[0],det_box[1]),(det_box[2],det_box[3]), colors[ix%len(colors)], 2)
        
        print('Frame {}'.format(idx))
        # writer.write(canvas)
        # cv2.imshow('track', canvas)
        # cv2.imshow('det',canvas_det)
        
        # if cv2.waitKey(1)==27:
        #     break

def experiment(dataset_obj, seqName, model, params=None, configs=None):
    dataset_obj.choice(seqName)
    targets=main(dataset_obj, model, params=params, configs=configs)
    base_dir='./track_results'
    hf=open(op.join(base_dir, '{}_H.txt'.format(seqName)), 'w')
    wf=open(op.join(base_dir, '{}_W.txt'.format(seqName)), 'w')
    lxf=open(op.join(base_dir, '{}_LX.txt'.format(seqName)), 'w')
    lyf=open(op.join(base_dir, '{}_LY.txt'.format(seqName)), 'w')

    for target in targets:
        pass
            

if __name__=='__main__':
    cfg.PHASE='TEST'
    cfg.NUM_CLASSES=len(CLASSES)
    cfg.TEST.RPN_PRE_NMS_TOP_N=5000
    cfg.TEST.RPN_POST_NMS_TOP_N=2000
    cfg.DET_SCORE_THRESH=0.95
    cfg.TRACK_SCORE_THRESH=0.01
    cfg.TRACK_MAX_DIST=20
    cfg.TEST.RPN_NMS_THRESH=0.7
    cfg.TEST.NMS_THRESH=0.7
    cfg.TEMP_NUM=5

    model_path='./ckpt/dl_mot_epoch_40.pkl'
    dataset_obj=detrac.Detrac(im_width=im_width, im_height=im_height, name='DETRAC', load_gt=False)
    dataset_obj.choice('MVI_39821')

    model=MotFRCNN(im_width, im_height, pretrained=False)
    
    params, configs=load_parameters(model, model_path=model_path)
    model.cuda()
    main(dataset_obj, model=model, params=params, configs=configs)
