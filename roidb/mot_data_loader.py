import numpy as np
import cv2
import os
import os.path as op
from rpn.template import get_template
import rpn.util as U
import rpn.generate_anchors as G
import roidb.box_utils as butil
import roidb.image_utils as util
from core.config import cfg

MOT_SUBDIRS = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
MAX_SAMPLES=-1
MAX_TEMPLATE_SIZE=cfg.TEMP_MAX_SIZE
DEBUG=True

class MOTDataLoader(object):
    def __init__(self, im_width, im_height, batch_size=8):
        self.mot_dir = '/mnt/sda7/MOT2017/train'
        self.img_dirs = []
        self.anno_dirs = []
        for DIR in MOT_SUBDIRS:
            img_dirs = op.join(DIR, 'img1')
            anno_dirs = op.join(DIR, 'gt')
            self.img_dirs.append(img_dirs)
            self.anno_dirs.append(anno_dirs)

        self.index = 0
        self.vis_dir = './vis_mot'
        self.vis_index = 0
        self.person_conf = 0.6

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

        assert len(self.img_dirs) == len(self.anno_dirs), 'Data and annotation dirs not uniformed: {} vs {}'.format(
            len(self.img_dirs), len(self.anno_dirs))

        self.num_sequences = len(self.img_dirs)
        self.num_images=0
        self.num_visualize = 100
        
        self.permute_inds = np.random.permutation(np.arange(self.num_sequences))

        self.im_w = im_width
        self.im_h = im_height

        self.bound=(im_width, im_height)
        self.out_size=(im_width//self.stride, im_height//self.stride)

        self.batch_size=batch_size

        self.max_interval=10 if cfg.PHASE=='TRAIN' else 1
        self.iter_stop=False
        
        self.gt_annotations = self.parse_gt()
        self.enum_sequences()
        
        self.templates=get_template(min_size=cfg.TEMP_MIN_SIZE, max_size=cfg.TEMP_MAX_SIZE, num_templates=cfg.TEMP_NUM)
        self.raw_anchors=G.generate_anchors(self.basic_size, self.ratios, self.scales)
        self.track_raw_anchors=G.generate_anchors(self.track_basic_size, self.track_ratios, self.track_scales)
        
    def wh2xy(self, bbox):
        return np.asarray([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

    def parse_gt(self):
        gt_annotations = {}
        for i in range(self.num_sequences):
            gt_file = op.join(self.mot_dir, self.anno_dirs[i], 'gt.txt')
            with open(gt_file, 'r') as f:
                lines = f.readlines()

            sample_subdir = {}
            for line in lines:
                items = line.rstrip().split(',')
                items = list(map(float, items))
                target_id = int(items[1]) - 1
                frame_id = int(items[0]) - 1
                cat_id = int(items[6])
                score = items[8]
                bbox = items[2:6]  # [x,y,w,h]
                if frame_id not in sample_subdir.keys():
                    sample_subdir[frame_id] = []
                if cat_id == 1 and score > self.person_conf:
                    sample = {}
                    sample['frame'] = frame_id
                    sample['id'] = target_id
                    sample['cat'] = cat_id
                    sample['score'] = score
                    sample['bbox'] = self.wh2xy(bbox)
                    sample_subdir[frame_id].append(sample)
            gt_annotations[MOT_SUBDIRS[i]] = sample_subdir
        return gt_annotations
    
    def enum_sequences(self):
        self.images_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
        self.index_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
        self.round_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
        for i in range(self.num_sequences):
            annotation=self.gt_annotations[MOT_SUBDIRS[i]]
            num_samples_this_dir=len(annotation)
            self.images_per_seq[i]=num_samples_this_dir
            self.num_images+=num_samples_this_dir
        self.upper_bound_per_seq=self.images_per_seq-self.batch_size-1
        
    def get_num_samples(self):
        return np.sum(self.images_per_seq)

    def clip_boxes(self, boxes):
        boxes[0]=np.maximum(0, boxes[0])
        boxes[1]=np.maximum(0, boxes[1])
        boxes[2]=np.minimum(self.im_w, boxes[2])-1
        boxes[3]=np.minimum(self.im_h, boxes[3])-1

    def get_minibatch(self):
        if self.iter_stop:
            self.iter_stop=False
            return None
        
        roidbs=[]
        index=self.permute_inds[self.index]
        
        while self.index_per_seq[index]>self.upper_bound_per_seq[index]:
            self.index+=1
            if self.index==self.num_sequences:
                self.index=0
            index=self.permute_inds[self.index]
            
        img_dir=op.join(self.mot_dir, MOT_SUBDIRS[index], 'img1') 
        annotation=self.gt_annotations[MOT_SUBDIRS[index]]
        
        img_files = sorted(os.listdir(img_dir))
        '''
        if MAX_SAMPLES>0:
            max_samples=min(MAX_SAMPLES, len(img_files))
            img_files=img_files[:max_samples]
        '''
        cur_image_index=self.index_per_seq[index]
        temp_inds=np.arange(cur_image_index, cur_image_index+self.batch_size)
        real_num_samples=temp_inds.size
        
        interval=np.random.randint(1, self.max_interval+1, size=real_num_samples)
        det_inds=temp_inds+interval
        det_inds=np.minimum(det_inds, self.images_per_seq[index]-1)
#        non_empty_batch=False
        
        for _, inds in enumerate(zip(temp_inds, det_inds)):
            roidb={}
            temp_boxes={}
            det_boxes={}
            
            temp_image=None
            det_image=None

            temp_gt_classes={}
            det_gt_classes={}
            
            for ind in inds:
                image=cv2.imread(op.join(img_dir, img_files[ind]))
                
                h,w=image.shape[0:2]
                image=cv2.resize(image, (self.im_w, self.im_h), interpolation=cv2.INTER_LINEAR)
                nh, nw=image.shape[0:2]

                yscale=1.0*nh/h
                xscale=1.0*nw/w

                objects = annotation[ind]
#                obj_boxes=np.zeros((0,4),dtype=np.float32)
                if len(objects)==0:
                    print('{} has no targets'.format(op.join(img_dir, img_files[ind])))
                    if DEBUG and self.vis_index < self.num_visualize:
                        cv2.imwrite(op.join('mot_no_target', img_files[ind]), image)
                        
                for obj in objects:
                    obj_id=obj['id']
                    bbox = obj['bbox'].copy()

                    bbox[[0,2]]*=xscale
                    bbox[[1,3]]*=yscale
                    self.clip_boxes(bbox)
    
                    bbox = bbox.astype(np.int32)
#                    obj_boxes=np.append(obj_boxes, bbox[np.newaxis,:], 0)
                    if ind==inds[0]:
                        temp_boxes[obj_id]=bbox
                        temp_gt_classes[obj_id]=1
                    else:
                        det_boxes[obj_id]=bbox
                        det_gt_classes[obj_id]=1
                        
#                    if self.vis_index < self.num_visualize:
#                        cv2.rectangle(image_cpy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                        
                if ind==inds[0] and len(temp_boxes.keys())>0:
                    temp_image=image[np.newaxis, :,:,:].astype(np.float32)
#                    non_empty_batch=True
                elif ind==inds[1] and len(temp_boxes.keys())>0:
                    det_image=image[np.newaxis, :,:,:].astype(np.float32)
                
            ref_boxes_align,det_boxes_align,ref_classes_align,det_classes_align=butil.align_boxes(temp_boxes, \
                det_boxes, temp_gt_classes, det_gt_classes)       
#            print(ref_boxes_align)
#            print(det_boxes_align)
            good_inds=[]

            if len(ref_boxes_align)>0:
                roidb['raw_temp_boxes']=ref_boxes_align
                roidb['temp_image']=temp_image
                roidb['det_image']=det_image
                roidb['temp_boxes']=ref_boxes_align
                roidb['det_boxes']=det_boxes_align
                roidb['temp_classes']=ref_classes_align
                roidb['det_classes']=det_classes_align
                
                '''NHWC'''
                bound=(det_image.shape[2], det_image.shape[1])
                search_boxes=np.zeros((0,4),dtype=np.float32)

                for i, box in enumerate(ref_boxes_align):
#                    if box[2]-box[0]<MAX_TEMPLATE_SIZE-1 and box[3]-box[1]<MAX_TEMPLATE_SIZE-1:
                    if cfg.PHASE=='TEST':
                        _,search_box=butil.best_search_box_test(self.templates, box, bound)
                    else:
                        search_box,_,_=butil.best_search_box_train(box, det_boxes_align[i], self.templates, self.track_raw_anchors, bound, self.TK, (self.rpn_conv_size, self.rpn_conv_size), cfg.TRAIN.TRACK_RPN_POSITIVE_THRESH)
                    search_boxes=np.append(search_boxes, search_box.reshape(1,-1), 0)
                    good_inds.append(i)
#                    else:
#                        search_boxes=np.append(search_boxes, np.array([[0,0,0,0]]), 0)
                roidb['search_boxes']=search_boxes
                roidb['bound']=bound
                roidb['good_inds']=good_inds

                track_anchors=G.gen_region_anchors(self.track_raw_anchors, search_boxes, bound, K=self.TK, size=(self.rpn_conv_size,self.rpn_conv_size)) 
                '''track anchors'''
                roidb['track_anchors']=track_anchors
                bbox_overlaps=U.bbox_overlaps_per_image(np.vstack(track_anchors), det_boxes_align, branch='rpn')
                roidb['bbox_overlaps']=bbox_overlaps                                                   
            
                dummy_search_box=np.array([[0,0,self.im_w-1,self.im_h-1]])
                anchors=G.gen_region_anchors(self.raw_anchors, dummy_search_box, bound, K=self.K, size=self.out_size)
                '''detection anchors'''
                roidb['anchors']=anchors[0]
                roidbs.append(roidb)           
        '''
        [NHWC,NHWC]
        '''
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
        
        '''
        Fucking MOT dataset with too many frames in which there's no target
        '''
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
    
if __name__=='__main__':
    loader=MOTDataLoader(100,100,1)
    print(loader.num_sequences)
    print(loader.get_num_samples())