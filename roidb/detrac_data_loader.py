import numpy as np
import cv2
import os
import os.path as op
#import shutil
import rpn.util as U
import rpn.generate_anchors as G
from rpn.template import get_template
import roidb.image_utils as util
import roidb.box_utils as butil
import xml.etree.ElementTree as ET
from core.config import cfg
import copy

MAX_SEQ_LEN=-1
MAX_TEMPLATE_SIZE=cfg.TEMP_MAX_SIZE
EXTRA_SEQS=['MVI_39761','MVI_39781','MVI_39811','MVI_39851','MVI_39931','MVI_40152','MVI_40162','MVI_40211','MVI_40213','MVI_40991','MVI_40992','MVI_63544']
CAT_IND_MAP={'car':1,'van':2,'bus':3,'truck':4}

class DetracDataLoader(object):
    def __init__(self, im_width, im_height, batch_size=8):
        self.data_dir = op.join(cfg.DATA_DIR,'Insight-MVT_Annotation_Train')
        self.anno_dir = op.join(cfg.DATA_DIR,'DETRAC-Train-Annotations-XML')
        
        self.stride=cfg.STRIDE
        self.basic_size=cfg.BASIC_SIZE
        self.ratios=cfg.RATIOS
        self.scales=cfg.SCALES
        
        self.track_basic_size=cfg.TRACK_BASIC_SIZE
        self.track_ratios=cfg.TRACK_RATIOS
        self.track_scales=cfg.TRACK_SCALES
        
        self.K=len(self.ratios)*len(self.scales)
        self.TK=len(self.track_ratios)*len(self.track_scales)
        self.rpn_conv_size=cfg.RPN_CONV_SIZE
        
        self.img_dirs=sorted(os.listdir(self.data_dir))
        self.anno_files=sorted(os.listdir(self.anno_dir))

        for ext_seq in EXTRA_SEQS:
            ext_anno='{}.xml'.format(ext_seq)
#            assert ext_anno in self.anno_files, '{} not exists'.format(ext_anno)
            self.anno_files.remove(ext_anno)
            self.img_dirs.remove(ext_seq)

        self.index = 0
        self.vis_dir = './vis_vid'
        self.vis_index = 0

        self.margin_gain=0.2
        
        self.im_w = im_width
        self.im_h = im_height

        self.bound=(im_width, im_height)
        self.out_size=(im_width//self.stride, im_height//self.stride)
        
        self.batch_size=batch_size
        self.roi_size=cfg.DET_ROI_SIZE-cfg.TEMP_ROI_SIZE+1

        self.num_sequences = len(self.anno_files)
        self.num_images=0
        
        self.num_visualize = 100
        self.permute_inds = np.random.permutation(np.arange(self.num_sequences))

        self.max_interval=4 if cfg.PHASE=='TRAIN' else 1
        
        self.iter_stop=False
        self.enum_sequences()
        
        self.templates=get_template(min_size=cfg.TEMP_MIN_SIZE, max_size=cfg.TEMP_MAX_SIZE, num_templates=cfg.TEMP_NUM)

        self.raw_anchors=G.generate_anchors(self.basic_size, self.ratios, self.scales)
        dummy_search_box=np.array([[0,0,self.im_w-1,self.im_h-1]])
        self.det_anchors=G.gen_region_anchors(self.raw_anchors, dummy_search_box, self.bound, K=self.K, size=self.out_size)[0]

        self.track_raw_anchors=G.generate_anchors(self.track_basic_size, self.track_ratios, self.track_scales)

    def gen_anchors(self, search_boxes, bound):
        box_anchors=[]
        for i in range(len(search_boxes)):
            A=self.roi_size**2
            K=self.K
            shifts_ctrs = G.calc_roi_align_shifts(search_boxes[i], self.roi_size, bound, stride=self.stride)
            anchors = self.raw_anchors.reshape((1, K, 4)) + shifts_ctrs.reshape((1, A, 4)).transpose((1, 0, 2))
            anchors = anchors.reshape((A*K, 4))
            box_anchors.append(anchors)
        
        return box_anchors
    
    def enum_sequences(self):
        self.images_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
        self.index_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
        for i in range(self.num_sequences):
            img_dir=self.img_dirs[i]
            num_samples_this_dir=len(os.listdir(op.join(self.data_dir,img_dir)))
            self.images_per_seq[i]=min(num_samples_this_dir, MAX_SEQ_LEN if MAX_SEQ_LEN>0 else num_samples_this_dir)
            self.num_images+=num_samples_this_dir
        self.upper_bound_per_seq=self.images_per_seq-self.batch_size-1

    def get_num_samples(self):
        return np.sum(self.images_per_seq)

    def get_minibatch(self):
        return self.get_minibatch_inter_img()

    def get_minibatch_inter_img(self):
        if self.iter_stop:
            self.iter_stop=False
            return None
        
        roidbs=[]
        index = self.permute_inds[self.index]
        while self.index_per_seq[index]>self.upper_bound_per_seq[index]:
            self.index+=1
            if self.index==self.num_sequences:
                self.index=0
            index=self.permute_inds[self.index]
            
        anno_file=op.join(self.anno_dir, self.anno_files[index])
        img_dir=op.join(self.data_dir, self.img_dirs[index])

        img_files=sorted(os.listdir(img_dir))
        tree = ET.parse(anno_file)
        root = tree.getroot()

        frames=root.findall('frame')

        cur_image_index=self.index_per_seq[index]
        ref_inds=np.arange(cur_image_index, cur_image_index+self.batch_size)
        
        real_num_samples=ref_inds.size
        interval=np.random.randint(1,self.max_interval+1, size=real_num_samples)
        det_inds=ref_inds+interval
        det_inds=np.minimum(det_inds, self.images_per_seq[index]-1)
        
        for _, inds in enumerate(zip(ref_inds, det_inds)):
            roidb={}
            
            ref_boxes={}
            det_boxes={}
            
            temp_image=None
            det_image=None

            temp_gt_classes={}
            det_gt_classes={}

            for ind in inds:
                img_file=img_files[ind]
                frame=frames[ind]
                
                image = cv2.imread(op.join(img_dir, img_file))
                
#                image, (xstart, ystart), scale=util.resize_and_pad_image(image, self.im_w, self.im_h)
                h,w=image.shape[0:2]
                image=cv2.resize(image, (self.im_w, self.im_h), interpolation=cv2.INTER_LINEAR)
                nh, nw=image.shape[0:2]

                yscale=1.0*nh/h
                xscale=1.0*nw/w

                target_list=frame.find('target_list')
                targets=target_list.findall('target')
                        
                for obj in targets:
                    attribs=obj.attrib
                    obj_id = int(attribs['id'])

                    bbox = np.zeros(4, dtype=np.float32)

                    bbox_attribs=obj.find('box').attrib
                    attribute_attribs=obj.find('attribute').attrib

                    left=float(bbox_attribs['left'])
                    top=float(bbox_attribs['top'])
                    width=float(bbox_attribs['width'])
                    height=float(bbox_attribs['height'])

                    bbox[0]=left
                    bbox[1]=top
                    bbox[2]=left+width-1
                    bbox[3]=top+height-1
                    '''
                    bbox*=scale
                    bbox[[0,2]]+=xstart
                    bbox[[1,3]]+=ystart
                    '''
                    bbox[[0,2]]*=xscale
                    bbox[[1,3]]*=yscale

                    bbox=butil.clip_bboxes(bbox[np.newaxis,:], self.im_w, self.im_h).squeeze()

                    cat=attribute_attribs['vehicle_type']
                    if cat=='others':
                        cat='truck'
                    cat_ind=CAT_IND_MAP[cat]

                    if ind==inds[0]:
                        ref_boxes[obj_id]=bbox
                        temp_gt_classes[obj_id]=cat_ind
                    else:
                        det_boxes[obj_id]=bbox
                        det_gt_classes[obj_id]=cat_ind

                if ind==inds[0] and len(ref_boxes.keys())>0:
                    temp_image=image[np.newaxis, :,:,:].astype(np.float32)
                elif ind==inds[1] and len(ref_boxes.keys())>0:
                    det_image=image[np.newaxis, :,:,:].astype(np.float32)
                
            ref_boxes_align,det_boxes_align,ref_classes_align,det_classes_align=butil.align_boxes(ref_boxes, det_boxes, temp_gt_classes, det_gt_classes)
            good_inds=[]
            if len(ref_boxes_align)>0:
                roidb['raw_temp_boxes']=ref_boxes_align
#                temp_boxes=util.compute_template_boxes(ref_boxes_align, (temp_image.shape[2], temp_image.shape[1]), gain=self.margin_gain, shape='same')
                roidb['temp_image']=temp_image
                roidb['det_image']=det_image
#                roidb['temp_boxes']=temp_boxes
                roidb['temp_boxes']=ref_boxes_align
                roidb['det_boxes']=det_boxes_align
                roidb['temp_classes']=ref_classes_align
                roidb['det_classes']=det_classes_align
                '''NHWC'''
                bound=(det_image.shape[2], det_image.shape[1])
                
                search_boxes=np.zeros((0,4),dtype=np.float32)
                
                num_fg_anchors=-1
                best_track_anchors=None
                best_ind=0
                
                for i, box in enumerate(ref_boxes_align):
                    if box[2]-box[0]<MAX_TEMPLATE_SIZE-1 and box[3]-box[1]<MAX_TEMPLATE_SIZE-1:
                        if cfg.PHASE=='TEST':
                            _,search_box=butil.best_search_box_test(self.templates, box, bound)
                        else:
                            search_box, max_overlap_num, best_anchors=butil.best_search_box_train(box, det_boxes_align[i], self.templates, self.track_raw_anchors, bound, self.TK, (self.rpn_conv_size, self.rpn_conv_size), cfg.TRAIN.TRACK_RPN_POSITIVE_THRESH)                            
                                                  
                            if max_overlap_num>num_fg_anchors:
                                num_fg_anchors=max_overlap_num
                                best_track_anchors=best_anchors
                                best_ind=i

                        search_boxes=np.append(search_boxes, search_box.reshape(1,-1), 0)
                        good_inds.append(i)
                    else:
                        search_boxes=np.append(search_boxes, np.array([[0,0,0,0]]), 0)

                if len(good_inds)>0:
                    roidb['search_boxes']=search_boxes
                    roidb['bound']=bound
                    roidb['good_inds']=np.array(good_inds, dtype=np.int32)

                    roidb['best_anchors']=best_track_anchors
                    roidb['best_ind']=best_ind                                        

                    '''detectio anchors'''
                    roidb['anchors']=self.det_anchors
                    roidbs.append(roidb)

        self.index_per_seq[index] += self.batch_size
        index_res=self.index_per_seq-self.upper_bound_per_seq
        index_res=index_res[self.permute_inds]
        valid_seq_inds=np.where(index_res<=0)[0]
        if valid_seq_inds.size==0:
            self.index_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
            self.round_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
            self.permute_inds = np.random.permutation(np.arange(self.num_sequences))
            self.index=0
            self.iter_stop=True
        else:    
            self.index+=1
            if self.index==self.num_sequences:
                self.index=0
        if len(roidbs)>0 and len(roidbs)<self.batch_size:
            top_n=len(roidbs)
            print('Pad roidbs with previous elements. From {} to {}'.format(top_n, self.batch_size))
            m=self.batch_size//len(roidbs)-1
            n=self.batch_size%len(roidbs)
            for i in range(m):
                roidbs.extend(roidbs[:top_n])
            if n>0:
                roidbs.extend(roidbs[:n])
            assert len(roidbs)==self.batch_size, 'roidbs length is not valid: {}/{}'.format(len(roidbs), self.batch_size)
        return roidbs