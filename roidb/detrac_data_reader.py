from .data_reader import DataReader
import numpy as np
import cv2
import os
import os.path as op
import rpn.util as U
import rpn.generate_anchors as G
from rpn.template import get_template
import roidb.image_utils as util
import roidb.box_utils as butil
import xml.etree.ElementTree as ET
from core.config import cfg
from time import time

EXTRA_SEQS=['MVI_39761','MVI_39781','MVI_39811','MVI_39851',\
            'MVI_39931','MVI_40152','MVI_40162','MVI_40211',\
            'MVI_40213','MVI_40991','MVI_40992','MVI_63544']
CAT_IND_MAP={'car':1,'van':2,'bus':3,'truck':4}

class DetracDataReader(DataReader):
    def __init__(self, im_width, im_height, batch_size=8):
        super(DetracDataReader, self).__init__(im_width, im_height, batch_size=batch_size)
        self.data_dir = op.join(cfg.DATA_DIR,'Insight-MVT_Annotation_Train')
        self.anno_dir = op.join(cfg.DATA_DIR,'DETRAC-Train-Annotations-XML')                        
                
        self.img_dirs=sorted(os.listdir(self.data_dir))
        self.anno_files=sorted(os.listdir(self.anno_dir))

        for ext_seq in EXTRA_SEQS:
            ext_anno='{}.xml'.format(ext_seq)
            self.anno_files.remove(ext_anno)
            self.img_dirs.remove(ext_seq)

        self.index = 0        
        self.num_sequences = len(self.anno_files)
        choice_inds=np.random.choice(np.arange(self.num_sequences), size=10, replace=False)
        self.anno_files=[self.anno_files[i] for i in choice_inds]
        self.img_dirs=[self.img_dirs[i] for i in choice_inds]
        self.num_sequences=choice_inds.size
        self.num_images=0
        
        self.num_visualize = 100
        self.permute_inds = np.random.permutation(np.arange(self.num_sequences))

        self.max_interval=5 if cfg.PHASE=='TRAIN' else 1
        
        self.iter_stop=False

        self.enum_sequences()     
        self._parse_all_anno()  

        cfg.NUM_CLASSES=len(CAT_IND_MAP)+1
        self.MAX_TEMPLATE_SIZE=cfg.TEMP_MAX_SIZE

    def enum_sequences(self):
        self.images_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
        self.index_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
        self.start_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
        for i in range(self.num_sequences):
            img_dir=self.img_dirs[i]
            num_samples_this_dir=len(os.listdir(op.join(self.data_dir,img_dir)))
            self.images_per_seq[i]=num_samples_this_dir
            self.start_per_seq[i]=self.num_images
            self.num_images+=num_samples_this_dir
        self.upper_bound_per_seq=self.images_per_seq-self.batch_size-1

    def _parse_all_anno(self):
        print('Preparing dataset...')
        all_anno=[]
        for i in range(self.num_sequences):
            anno_file=op.join(self.anno_dir, self.anno_files[i])
            tree = ET.parse(anno_file)
            root = tree.getroot()

            frames=root.findall('frame')

            for frame in frames:
                target_list=frame.find('target_list')
                targets=target_list.findall('target')

                all_anno.append(targets)

        assert self.num_images==len(all_anno), 'Annotations: {} and num_images: {}'.format(len(all_anno), self.num_images)
        print('Dataset is available')
        self.annotations=all_anno
    

    def overlap_list2array(self, num_anchors, overlaps_list):
        n_boxes=len(overlaps_list)
        overlaps=np.zeros((n_boxes*num_anchors, n_boxes))
        for i in range(n_boxes):
            overlaps[i*num_anchors:(i+1)*num_anchors,i][...]=overlaps_list[i].ravel()
        return overlaps

    def _get_roidb(self, seq_index, ref_ind, det_ind):
        img_dir=op.join(self.data_dir, self.img_dirs[seq_index])

        img_files=sorted(os.listdir(img_dir))
        roidb={}
            
        temp_boxes={}
        det_boxes={}

        temp_boxes4det=np.zeros((0,4),dtype=np.float32)
        det_boxes4det=np.zeros((0,4),dtype=np.float32)
        
        temp_image=None
        det_image=None

        temp_gt_classes=np.zeros(0,dtype=np.int32)
        det_gt_classes=np.zeros(0,dtype=np.int32)

        for ind in [ref_ind, det_ind]:
            img_file=img_files[ind]
            #HWC
            image = cv2.imread(op.join(img_dir, img_file))
            h,w=image.shape[0:2]

            image=cv2.resize(image, (self.im_w, self.im_h), interpolation=cv2.INTER_LINEAR)
            nh, nw=image.shape[0:2]

            yscale=1.0*nh/h
            xscale=1.0*nw/w
                        
            targets=self.annotations[self.start_per_seq[seq_index]+ind]
            for obj in targets:
                    attribs=obj.attrib
                    obj_id = int(attribs['id'])

                    bbox = np.zeros(4, dtype=np.float32)

                    bbox_attribs=obj.find('box').attrib
                    attribute_attribs=obj.find('attribute').attrib
                    cat=attribute_attribs['vehicle_type']
                    if cat=='others':
                        cat='truck'
                    cat_ind=CAT_IND_MAP[cat]

                    left=float(bbox_attribs['left'])
                    top=float(bbox_attribs['top'])
                    width=float(bbox_attribs['width'])
                    height=float(bbox_attribs['height'])

                    bbox[0]=left
                    bbox[1]=top
                    bbox[2]=left+width-1
                    bbox[3]=top+height-1

                    bbox[[0,2]]*=xscale
                    bbox[[1,3]]*=yscale

                    bbox=butil.clip_bboxes(bbox.reshape(1,4), self.im_w, self.im_h)

                    if ind==ref_ind:
                        temp_boxes[obj_id]=bbox
                        temp_boxes4det=np.append(temp_boxes4det, bbox.reshape(1,4), 0)
                        temp_gt_classes=np.append(temp_gt_classes, cat_ind)
                    else:
                        det_boxes[obj_id]=bbox
                        det_boxes4det=np.append(det_boxes4det, bbox.reshape(1,4), 0)   
                        det_gt_classes=np.append(det_gt_classes, cat_ind)                    

            if ind==ref_ind and len(temp_boxes.keys())>0:
                temp_image=image[np.newaxis, :,:,:].astype(np.float32)
            elif ind==det_ind and len(temp_boxes.keys())>0:
                det_image=image[np.newaxis, :,:,:].astype(np.float32)

        temp_boxes_align,det_boxes_align = butil.align_boxes(temp_boxes, det_boxes)
        good_inds=[]
        if len(temp_boxes_align)>0:            
            '''NHWC'''            
            search_boxes=np.zeros((0,4),dtype=np.float32)
            num_instances=0
            overlaps_list=[]
            anchors_list=[]
            for i, box in enumerate(temp_boxes_align):
                temp_box_ok=False
                if box[2]-box[0]<self.MAX_TEMPLATE_SIZE-1 and box[3]-box[1]<self.MAX_TEMPLATE_SIZE-1:
                    if cfg.PHASE=='TEST':
                        _,search_box=butil.best_search_box_test(self.templates, box, self.bound)
                    else:
                        ret=butil.best_search_box_random(box, \
                             det_boxes_align[i], self.templates, self.template_anchors, \
                                 self.bound, cfg.TRAIN.TRACK_RPN_POSITIVE_THRESH)
                        if ret is not None:
                            search_box, overlaps, anchors= ret
                            num_instances+=1
                            
                            overlaps_list.append(overlaps)
                            anchors_list.append(anchors)
                            search_boxes=np.append(search_boxes, search_box.reshape(1,-1), 0)
                            good_inds.append(i)
                            temp_box_ok=True
                            if num_instances==5:
                                break

                if not temp_box_ok:
                    search_boxes=np.append(search_boxes, np.array([[0,0,0,0]]), 0)

        if len(good_inds)>0:  
            good_inds=np.array(good_inds, dtype=np.int32)              
            roidb['temp_image']=temp_image
            roidb['det_image']=det_image
            roidb['temp_boxes']=temp_boxes_align[good_inds]
            roidb['det_boxes']=det_boxes_align[good_inds]
            roidb['temp_classes']=temp_gt_classes
            roidb['det_classes']=det_gt_classes
            roidb['temp_boxes4det']=temp_boxes4det
            roidb['det_boxes4det']=det_boxes4det

            roidb['search_boxes']=search_boxes[good_inds]
            roidb['bound']=self.bound
            roidb['bbox_overlaps']=self.overlap_list2array(self.template_anchors[0].shape[0], overlaps_list)

            roidb['track_anchors']=anchors_list

            '''detectio anchors'''
            roidb['anchors']=self.det_anchors

        return roidb
        
