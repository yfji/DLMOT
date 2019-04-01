from .dataset import Dataset
import roidb.image_utils as util
import os
import os.path as op
import numpy as np
import cv2

class KITTI(object):
    def __init__(self, im_width=0, im_height=0, name='Detrac'):
        super(KITTI, self).__init__(im_width=im_width, im_height=im_height, name=name)
        print('Using benchmark {}'.format(self.dataset_name))
        self.data_dir = '/mnt/sda7/DETRAC-train-data/Insight-MVT_Annotation_Train'