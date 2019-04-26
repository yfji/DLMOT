import os.path as op
import numpy as np
import cv2
from model.mot_forward import MotFRCNN
from core.config import cfg
from dataset import detrac,vid,vot
from dataset.data_loader import DataLoader
from inference import CLASSES, inference_detect
import roidb.box_utils as butil
import rpn.generate_anchors as G

im_width=640
im_height=384

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

K=len(cfg.RATIOS)*len(cfg.SCALES)
bound=(im_width, im_height)
out_size=(im_width//8, im_height//8)
det_raw_anchors=G.generate_anchors(cfg.BASIC_SIZE, cfg.RATIOS, cfg.SCALES)
dummy_search_box=np.array([[0,0,im_width-1,im_height-1]])
det_anchors=G.gen_region_anchors(det_raw_anchors, dummy_search_box, bound, K=K, size=out_size)[0]
bound=(im_width, im_height)

def main(dataset_obj, model=None):
    loader=DataLoader(dataset_obj)
#    video_path='./result.avi'
#    writer=cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (im_width, im_height))

    roidb={'anchors':det_anchors, 'bound':bound}

    for _, image in enumerate(loader):
        canvas=image.copy()
        roidb['temp_image']=image[np.newaxis,:,:,:].astype(np.float32, copy=True)
        roidb['det_image']=image[np.newaxis,:,:,:].astype(np.float32, copy=True)

        outputs=inference_detect(model, roidb, batch_size=1)

        output=outputs[0]
        cls_boxes=output['cls_bboxes']

        index=0
        for i in range(1, len(CLASSES)):
            bboxes=cls_boxes[i]
            for bbox in bboxes:
                bbox=bbox.astype(np.int32)
                cv2.rectangle(canvas, (bbox[0],bbox[1]),(bbox[2],bbox[3]), colors[index%len(colors)], 2)
                index+=1
        cv2.imshow('det', canvas)
        if cv2.waitKey(1)==27:
            break

if __name__=='__main__':
    model_path='./ckpt/dl_mot_epoch_6.pkl'
    cfg.PHASE='TEST'
    cfg.DET_SCORE_THRESH=0.95
    cfg.TEST.RPN_NMS_THRESH=0.8
    cfg.TEST.NMS_THRESH=0.6
    cfg.TEST.RPN_POST_NMS_TOP_N=1000
    cfg.TEST.DATASET='detrac'
    dataset_obj=detrac.Detrac(im_width=im_width, im_height=im_height, name='Detrac',load_gt=False)
    dataset_obj.choice('MVI_39811')
    model=MotFRCNN(im_width, im_height, pretrained=False)
    model.load_weights(model_path)
    model.cuda()
    
    main(dataset_obj, model=model)
