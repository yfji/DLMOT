import os.path as op
import numpy as np
import cv2
from model.mot_forward import MotFRCNN
from core.config import cfg
from dataset import detrac,vid,vot,mot
from dataset.data_loader import DataLoader
from inference import inference_track
import roidb.box_utils as butil
import rpn.generate_anchors as G
from rpn.template import get_template

im_width=640
im_height=384

cfg.TEMP_MIN_SIZE=64
cfg.TEMP_MAX_SIZE=256
cfg.TEMP_NUM=5
MAX_TEMPLATE_SIZE=cfg.TEMP_MAX_SIZE

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

templates=get_template(min_size=64, max_size=cfg.MAX_TEMPLATE_SIZE, num_templates=cfg.TEMP_NUM)
det_raw_anchors=G.generate_anchors(cfg.BASIC_SIZE, cfg.RATIOS, cfg.SCALES)
track_raw_anchors=G.generate_anchors(cfg.TRACK_BASIC_SIZE, cfg.TRACK_RATIOS, cfg.TRACK_SCALES)
K=len(cfg.RATIOS)*len(cfg.SCALES)
TK=len(cfg.TRACK_RATIOS)*len(cfg.TRACK_SCALES)
rpn_conv_size=cfg.RPN_CONV_SIZE
out_size=(im_width//8, im_height//8)

def add_new_targets(ids, boxes, new_ids, new_boxes):
    combined_ids=ids.copy()
    combined_boxes=boxes.copy()
    num_instances=len(ids)
    num_new_instances=new_boxes.shape[0]
    
    index=0
    new_index=0
    for i in range(num_instances):
        if combined_ids[i]==0 and new_index<num_new_instances:
            combined_ids[i]=1    #the ith target is lost and replaced with a new target in its position
            combined_boxes=np.insert(combined_boxes, index, new_boxes[new_index], axis=0)
            new_index+=1
            index+=1
        else:
            index+=1
    if new_index<len(new_boxes):
        combined_ids=np.append(combined_ids, np.ones(num_new_instances-new_index))
        combined_boxes=np.append(combined_boxes, new_boxes[new_index:,:], 0)
    
    return combined_ids, combined_boxes

def check_boxes(boxes):
    x1,y1,x2,y2=np.split(boxes, 4, axis=1)
    ws,hs=x2-x1+1,y2-y1+1
    good_inds=np.where(np.bitwise_and(ws<MAX_TEMPLATE_SIZE-1, hs<MAX_TEMPLATE_SIZE-1)==1)[0]
    return good_inds

def main(dataset_obj, model=None):
    loader=DataLoader(dataset_obj)

    temp_boxes=None
    temp_image=None
    det_image=None
    
    started=False
    track_started=False
    num_instances=0
    track_ids=None
    
    VIS_DATASET=False
    TRACK_LAST_FRAME=False

#    writer=cv2.VideoWriter('./track.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (im_width, im_height))

    frame_ind=0
    for _, (image, gt_boxes) in enumerate(loader):
        canvas=image.copy()
        if VIS_DATASET:
            if gt_boxes is not None:
                temp_boxes=np.asarray(gt_boxes[:4]).reshape(-1,4)
            for i, box in enumerate(temp_boxes):
                box=box.astype(np.int32)
                cv2.rectangle(canvas, (box[0],box[1]), (box[2],box[3]), tuple(colors[i]), 2)
        else:
            image=image.astype(np.float32)
            if len(gt_boxes)==0 and not started:
                print('Waiting for a frame with gt_box')
                continue
            started=True
            if not track_started:
                temp_boxes=np.asarray(gt_boxes[:,:4]).reshape(-1,4).astype(np.float32)
                temp_image=image.copy()
                for i, box in enumerate(gt_boxes):
                    box=box.astype(np.int32)
                    cv2.rectangle(canvas, (box[0],box[1]), (box[2],box[3]), tuple(colors[i]), 2)
                track_started=True
                num_instances=temp_boxes.shape[0]
                track_ids=np.ones(num_instances, dtype=np.int32)
            else:
                '''replace with detection'''                
                if not TRACK_LAST_FRAME and len(gt_boxes)>num_instances:
                    track_ids, temp_boxes=add_new_targets(track_ids, temp_boxes, np.arange(num_instances, len(gt_boxes)), gt_boxes[num_instances:])
                    print('add {} new targets'.format(len(gt_boxes)-num_instances))
                    num_instances=len(track_ids)
                
                if TRACK_LAST_FRAME:
                    num_instances=len(temp_boxes)

                det_image=image.copy()
                roidb=dict()
                bound=(temp_image.shape[1], temp_image.shape[0])
                roidb['bound']=bound
                roidb['temp_image']=temp_image[np.newaxis,:,:,:].astype(np.float32)
                roidb['det_image']=det_image[np.newaxis,:,:,:].astype(np.float32)
                roidb['good_inds']=check_boxes(temp_boxes)
                roidb['temp_boxes']=temp_boxes
                search_boxes=[]
                for temp_box in temp_boxes:
                    _,best_template=butil.best_search_box_test(templates, temp_box, bound)
                    search_boxes.append(best_template)
                
                search_boxes=np.array(search_boxes)
#                print(search_boxes)
                roidb['det_boxes']=None
                roidb['det_classes']=None
                roidb['temp_classes']=None
                roidb['search_boxes']=search_boxes

#                anchors=G.gen_region_anchors(track_raw_anchors, search_boxes, bound, K=TK, size=(rpn_conv_size, rpn_conv_size))[0]
#                roidb['track_anchors']=anchors
                dummy_search_box=np.array([[0,0,im_width-1,im_height-1]])
                anchors=G.gen_region_anchors(det_raw_anchors, dummy_search_box, bound, K=K, size=out_size)[0]
                roidb['anchors']=anchors

                bboxes_list=inference_track(model, roidb)
                last_temp_boxes=temp_boxes.copy()                

                if not TRACK_LAST_FRAME and len(bboxes_list)>0:
                    valid_boxes=np.zeros((0,4), dtype=np.float32)
                    for i, bboxes in enumerate(bboxes_list):
                        if len(bboxes)>0:
                            if len(bboxes.shape)>0:
                                bboxes=bboxes.squeeze()
                            bbox=bboxes[:4]
                            valid_boxes=np.append(valid_boxes, bbox.reshape(1,-1), 0)
                    if valid_boxes.shape[0]>0:
                        temp_boxes=valid_boxes.copy()
                
                index=0
                for i in range(num_instances):
                    if not track_ids[i]:
                        continue
                    boxes_one_target = bboxes_list[index]
                    if len(boxes_one_target)==0:
                        track_ids[i]=0
                        print('target {} disappears'.format(i))
                    else:
                        box_inst=bboxes_list[index][0].astype(np.int32)
                        cv2.rectangle(canvas, (box_inst[0], box_inst[1]), (box_inst[2],box_inst[3]), tuple(colors[i%len(colors)]), 2)
                    index+=1
                num_exist_instances=len(np.where(track_ids!=0)[0])
                for i in range(num_exist_instances):
                    temp_box=last_temp_boxes[i].astype(np.int32)
                    cv2.rectangle(canvas, (temp_box[0],temp_box[1]),(temp_box[2],temp_box[3]), (0,0,255), 1)
                if TRACK_LAST_FRAME:
                    temp_boxes=np.asarray(gt_boxes[:,:4]).reshape(-1,4).astype(np.float32)
                temp_image=image.copy()

        frame_ind+=1
        print('Frame {}'.format(frame_ind))
        cv2.imshow('benchmark', canvas)
#        writer.write(canvas)
        if cv2.waitKey(1)==27:
            break

def test_insert():
    ids=np.asarray([1,0,0,1,1,1])
    temp_boxes=np.ones((4,4))
    new_ids=np.ones(3)
    new_boxes=2*np.ones((3,4))
    
    combined_ids, combined_boxes=add_new_targets(ids, temp_boxes, new_ids, new_boxes)
    print(combined_ids)
    print(combined_boxes)
    
if __name__=='__main__':
    model_path='./ckpt/dl_mot_epoch_2.pkl'
    cfg.PHASE='TEST'
    cfg.TRACK_SCORE_THRESH=0.1
    cfg.TRACK_MAX_DIST=30
    dataset_obj=detrac.Detrac(im_width=im_width, im_height=im_height, name='Detrac')
    dataset_obj.choice('MVI_40201')

    model=MotFRCNN(im_width, im_height, pretrained=False)
    model.load_weights(model_path)
    model.cuda()
    
    main(dataset_obj, model=model)