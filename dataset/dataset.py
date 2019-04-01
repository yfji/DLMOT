import numpy as np
import cv2
import roidb.image_utils as util
from core.config import cfg
import rpn.generate_anchors as G

class Dataset(object):
    def __init__(self, im_width=0, im_height=0, name=None):
        self.im_w=im_width
        self.im_h=im_height
        self.dataset_name=name
        self.basic_size=cfg.BASIC_SIZE
        self.ratios=cfg.RATIOS
        self.scales=cfg.SCALES

        self.raw_anchors=G.generate_anchors(self.basic_size, self.ratios, self.scales)

    def get_dataset(self):
        raise NotImplementedError
        
    def __len__(self):
        return 0
    
    def __getitem__(self):
        raise NotImplementedError

    def gen_anchors(self, search_boxes, bound):
        stride=cfg.STRIDE
        rpn_conv_size=cfg.DET_ROI_SIZE-cfg.TEMP_ROI_SIZE+1
        K=len(cfg.RATIOS)*len(cfg.SCALES)
        box_anchors=G.gen_region_anchors(self.raw_anchors, search_boxes, bound, stride=stride, K=K, rpn_conv_size=rpn_conv_size)
        return box_anchors
        
    def imresize(self, image, boxes):
        if boxes is None:
            image=cv2.resize(image, (self.im_w, self.im_h), interpolation=cv2.INTER_LINEAR)
            return image, None        
        h,w=image.shape[:2]
        image=cv2.resize(image, (self.im_w, self.im_h), interpolation=cv2.INTER_LINEAR)
        nh,nw=image.shape[:2]
        xscale=1.0*nw/w
        yscale=1.0*nh/h
        boxes_scaled=np.zeros((0,4),dtype=np.float32)
        if len(boxes)>0:
            boxes_scaled=boxes.astype(np.float32, copy=True)
            boxes_scaled[:,[0,2]]*=xscale
            boxes_scaled[:,[1,3]]*=yscale
        return image, boxes_scaled
        '''
        image_pad,(start_x, start_y),scale = util.resize_and_pad_image(image, self.im_w, self.im_h)
        boxes_scaled=boxes.copy()
        if len(boxes)>0:
            boxes_scaled*=scale
            boxes_scaled[:,[0,2]]+=start_x
            boxes_scaled[:,[1,3]]+=start_y
        return image_pad, boxes_scaled
        '''