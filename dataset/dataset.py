import numpy as np
import cv2
import roidb.image_utils as util
from core.config import cfg
import rpn.generate_anchors as G

class Dataset(object):
    def __init__(self, im_width=0, im_height=0, name=None, load_gt=True):
        self.im_w=im_width
        self.im_h=im_height
        self.dataset_name=name
        self.basic_size=cfg.BASIC_SIZE
        self.ratios=cfg.RATIOS
        self.scales=cfg.SCALES
        self.load_gt=load_gt

        self.raw_anchors=G.generate_anchors(self.basic_size, self.ratios, self.scales)

    def get_dataset(self):
        raise NotImplementedError
        
    def __len__(self):
        return 0
    
    def __getitem__(self):
        raise NotImplementedError

        
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