import numpy as np
import cv2
import os
import os.path as op
#import shutil
import rpn.util as U
import rpn.generate_anchors as G
import roidb.image_utils as util
import xml.etree.ElementTree as ET
from core.config import cfg
import copy

#VID_SUBDIRS=['ILSVRC2015_VID_train_0001','ILSVRC2015_VID_train_0002','ILSVRC2015_VID_train_0003']
VID_SUBDIRS=['ILSVRC2015_VID_train_0003']

DEBUG=True
MAX_SEQ_LEN=-1

class VIDDataLoader(object):
    def __init__(self, im_width, im_height, batch_size=8):
        self.vid_dir = '/mnt/sda7/ILSVRC2015/Data/VID/train'
        self.annot_dir = '/mnt/sda7/VID_Manual/Annotations/train/random'
        self.img_dirs = []
        self.anno_dirs = []
        
        self.stride=cfg.STRIDE
        self.basic_size=cfg.BASIC_SIZE
        self.ratios=cfg.RATIOS
        self.scales=cfg.SCALES
        
        self.K=len(self.ratios)*len(self.scales)
        
        for DIR in VID_SUBDIRS:
            img_dirs = sorted(os.listdir(op.join(self.vid_dir, DIR)))
            anno_dirs = sorted(os.listdir(op.join(self.annot_dir, DIR)))
#            img_dirs = [op.join(DIR, _dir) for _dir in img_dirs]
            '''depend on anno dirs, not img dirs'''
            img_dirs=[op.join(DIR, _dir) for _dir in anno_dirs]
            anno_dirs = [op.join(DIR, _dir) for _dir in anno_dirs]
            self.img_dirs.extend(img_dirs)
            self.anno_dirs.extend(anno_dirs)
            
        self.index = 0
        self.vis_dir = './vis_vid'
        self.vis_index = 0

        self.margin_gain=0.2
        
        self.im_w = im_width
        self.im_h = im_height
        
        self.batch_size=batch_size
        self.roi_size=cfg.DET_ROI_SIZE-cfg.TEMP_ROI_SIZE+1
        '''INTER_SEQ v.s. INTER_IMG'''
        self.method=cfg.DATA_LOADER_METHOD
#        assert len(self.img_dirs) == len(self.anno_dirs), 'Data and annotation dirs not uniformed'

        self.num_sequences = len(self.anno_dirs)
        self.num_images=0
        
        self.num_visualize = 100
        self.permute_inds = np.random.permutation(np.arange(self.num_sequences))

        self.max_interval=50 if cfg.PHASE=='TRAIN' else 20
#        self.valid_seq_inds=np.zeros(0, dtype=np.int32)
        
        self.iter_stop=False
        self.enum_sequences()
        
        self.raw_anchors=G.generate_anchors(self.basic_size, self.ratios, self.scales)
 
    def xy2wh(self, boxes):
        x=boxes[:,0]
        y=boxes[:,1]
        w=boxes[:,2]-x+1
        h=boxes[:,3]-y+1
        return x,y,w,h
    
    def gen_anchors(self, search_boxes, bound):
        xs,ys,ws,hs=self.xy2wh(search_boxes)
        
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
        self.round_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
        for i in range(self.num_sequences):
            anno_dir = op.join(self.annot_dir, self.anno_dirs[i])
            
            num_samples_this_dir=len(os.listdir(anno_dir))
#            img_dir=op.join(self.vid_dir, self.anno_dirs[i])
#            num_imgs=len(img_dir)
#            assert num_imgs==num_samples_this_dir, 'Not uniformed :{}'.format(self.anno_dirs[i])
 
            self.images_per_seq[i]=min(num_samples_this_dir, MAX_SEQ_LEN if MAX_SEQ_LEN>0 else num_samples_this_dir)
            self.num_images+=num_samples_this_dir
#        self.upper_bound_per_seq=self.images_per_seq-self.batch_size-self.max_interval
        self.upper_bound_per_seq=self.images_per_seq-self.batch_size-1

    def get_num_samples(self):
        return np.sum(self.images_per_seq)

    def wh2xy(self, bbox):
        return np.asarray([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

    def filter_boxes(self, boxes):
        x1=boxes[:,0]
        y1=boxes[:,1]
        x2=boxes[:,2]
        y2=boxes[:,3]

        ws=x2-x1+1
        hs=y2-y1+1

        filter_inds=np.where(np.bitwise_and(ws>16,hs>16)==1)[0]
        return filter_inds

    def align_boxes(self, ref_boxes, det_boxes):
        ref_ids=list(ref_boxes.keys())
        det_ids=list(det_boxes.keys())

        ref_boxes_align = []
        det_boxes_align = []

        if len(ref_ids)==0:
            print('Empty ref_ids')

        else:
            ref_ids=np.asarray(sorted(ref_ids))
            det_ids=np.asarray(sorted(det_ids))

            for ref_id in ref_ids:
                ref_boxes_align.append(ref_boxes[ref_id])
                if ref_id not in det_ids:
                    det_boxes_align.append(np.zeros(4, dtype=np.float32))
                else:
                    det_boxes_align.append(det_boxes[ref_id])

        if len(ref_boxes_align)==0:
            ref_boxes_align=np.zeros((0,4))
        else:
            ref_boxes_align = np.vstack(ref_boxes_align)
        if len(det_boxes_align)==0:
            det_boxes_align = np.zeros((0, 4))
        else:
            det_boxes_align=np.vstack(det_boxes_align)
        
        if len(ref_ids)>0:
            inds=self.filter_boxes(ref_boxes_align)
            ref_boxes_align=ref_boxes_align[inds]
            det_boxes_align=det_boxes_align[inds]
        return ref_boxes_align, det_boxes_align
    
    def get_minibatch(self):
        if self.method=='INTER_IMG':
            return self.get_minibatch_inter_img()
        elif self.method=='INTER_SEQ':
            return self.get_minibatch_inter_seq()
        else:
            raise NotImplementedError

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
            
        anno_dir=op.join(self.annot_dir, self.anno_dirs[index])
        img_dir=op.join(self.vid_dir, self.anno_dirs[index])

        img_files=sorted(os.listdir(img_dir))
        anno_files=sorted(os.listdir(anno_dir))

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
            for ind in inds:
                img_file=img_files[ind]
                anno_file=anno_files[ind]
                img_index=img_file[:img_file.rfind('.')]
                anno_index=anno_file[:anno_file.rfind('.')]
                assert img_index==anno_index, 'Index not uniformed'
                
                image = cv2.imread(op.join(img_dir, img_file))
                
#                image, (xstart, ystart), scale=util.resize_and_pad_image(image, self.im_w, self.im_h)
                h,w=image.shape[0:2]
                image=cv2.resize(image, (self.im_w, self.im_h), interpolation=cv2.INTER_LINEAR)
                nh, nw=image.shape[0:2]

                yscale=1.0*nh/h
                xscale=1.0*nw/w

                xml_file = op.join(anno_dir, anno_files[ind])
                tree = ET.parse(xml_file)
                root = tree.getroot()
    
                objects = root.findall('object')

                if len(objects)==0:
#                    print('{} has no targets'.format(op.join(img_dirs[batch_index], img_files[batch_index][ind])))
                    if DEBUG and self.vis_index < self.num_visualize:
                        cv2.imwrite(op.join('mot_no_target', img_files[ind]), image)
                        
                for obj in objects:
                    obj_id = int(obj.find('trackid').text)
                    bndbox = obj.find('bndbox')
                    
                    bbox = np.zeros(4, dtype=np.float32)
                    bbox[0]=float(bndbox.find('xmin').text)
                    bbox[1]=float(bndbox.find('ymin').text)
                    bbox[2]=float(bndbox.find('xmax').text)
                    bbox[3]=float(bndbox.find('ymax').text)
                    '''
                    bbox*=scale
                    bbox[[0,2]]+=xstart
                    bbox[[1,3]]+=ystart
                    '''
                    bbox[[0,2]]*=xscale
                    bbox[[1,3]]*=yscale

                    if ind==inds[0]:
                        ref_boxes[obj_id]=bbox
                    else:
                        det_boxes[obj_id]=bbox
                        
                if ind==inds[0] and len(ref_boxes.keys())>0:
                    temp_image=image[np.newaxis, :,:,:].astype(np.float32)
                elif ind==inds[1] and len(ref_boxes.keys())>0:
                    det_image=image[np.newaxis, :,:,:].astype(np.float32)
                
            ref_boxes_align, det_boxes_align=self.align_boxes(ref_boxes, det_boxes)
            
            if len(ref_boxes_align)>0:
                roidb['raw_temp_boxes']=ref_boxes_align
#                temp_boxes=util.compute_template_boxes(ref_boxes_align, (temp_image.shape[2], temp_image.shape[1]), gain=self.margin_gain, shape='same')
                roidb['temp_image']=temp_image
                roidb['det_image']=det_image
#                roidb['temp_boxes']=temp_boxes
                roidb['temp_boxes']=ref_boxes_align
                roidb['det_boxes']=det_boxes_align
                
                '''NHWC'''
                bound=(det_image.shape[2], det_image.shape[1])
                search_boxes=util.calc_search_boxes(ref_boxes_align, bound)
                roidb['search_boxes']=search_boxes
                roidb['bound']=bound
                '''
                Next
                roidb['bbox_overlaps']
                '''
#                anchors=self.gen_anchors(search_boxes, bound)
                anchors=G.gen_region_anchors(self.raw_anchors, search_boxes, bound, stride=self.stride, K=self.K, rpn_conv_size=self.roi_size)
                bbox_overlaps=U.bbox_overlaps_per_image(np.vstack(anchors), det_boxes_align)
                roidb['anchors']=anchors
                roidb['bbox_overlaps']=bbox_overlaps
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
        
        return roidbs
    
    def get_minibatch_inter_seq(self):
        if self.iter_stop:
            self.iter_stop=False
            return None
        
        roidbs=[]
        
        index_res=self.index_per_seq-self.upper_bound_per_seq
        '''permute'''
        index_res=index_res[self.permute_inds]
        valid_seq_inds=np.where(index_res<=0)[0]
        num_valid_seqs=len(valid_seq_inds)
        subdir_inds=[self.permute_inds[valid_seq_inds[i%num_valid_seqs]] for i in range(self.index, self.index+self.batch_size)]
        
        chosen_samples_per_seq={}
        for i in subdir_inds:
            if i not in chosen_samples_per_seq.keys():
                chosen_samples_per_seq[i]=0
            chosen_samples_per_seq[i]+=1

        img_dirs=[]
        anno_dirs=[]
        img_files=[]
        anno_files=[]
        cur_image_inds=[]
        
        for i in subdir_inds:
            '''already permuted'''
#            img_dirs.append(op.join(self.vid_dir, self.img_dirs[i]))
            img_dirs.append(op.join(self.vid_dir, self.anno_dirs[i]))
            anno_dirs.append(op.join(self.annot_dir, self.anno_dirs[i]))
            
            img_files.append(sorted(os.listdir(img_dirs[-1])))
            anno_files.append(sorted(os.listdir(anno_dirs[-1])))
            
            cur_image_inds.append(self.index_per_seq[i])
            
        ref_inds=np.asarray(cur_image_inds)
        tmp=copy.deepcopy(chosen_samples_per_seq)
        for i in range(len(ref_inds)):
            seq_ind=subdir_inds[i]
            ref_inds[i]+=(tmp[seq_ind]-chosen_samples_per_seq[seq_ind])
            chosen_samples_per_seq[seq_ind]-=1

        real_num_samples=ref_inds.size
        interval=np.random.randint(1,self.max_interval+1, size=real_num_samples)
        det_inds=ref_inds+interval
        det_inds=np.minimum(det_inds, self.images_per_seq[subdir_inds]-1)
        
        non_empty_batch=False
        for batch_index, inds in enumerate(zip(ref_inds, det_inds)):
            if len(img_files[batch_index])==0:
                continue
            
            roidb={}
            
            ref_boxes={}
            det_boxes={}
            
            temp_image=None
            det_image=None
            
            for ind in inds:
                img_file=img_files[batch_index][ind]
                anno_file=anno_files[batch_index][ind]
                img_index=img_file[:img_file.rfind('.')]
                anno_index=anno_file[:anno_file.rfind('.')]
                assert img_index==anno_index, 'Index not uniformed'
                
                image = cv2.imread(op.join(img_dirs[batch_index], img_files[batch_index][ind]))

                image, (xstart, ystart), scale=util.resize_and_pad_image(image, self.im_w, self.im_h)
    
                xml_file = op.join(anno_dirs[batch_index], anno_files[batch_index][ind])
                tree = ET.parse(xml_file)
                root = tree.getroot()
    
                objects = root.findall('object')

                if len(objects)==0:
#                    print('{} has no targets'.format(op.join(img_dirs[batch_index], img_files[batch_index][ind])))
                    if DEBUG and self.vis_index < self.num_visualize:
                        cv2.imwrite(op.join('mot_no_target', img_files[batch_index][ind]), image)
                        
                for obj in objects:
                    obj_id = int(obj.find('trackid').text)
                    bndbox = obj.find('bndbox')
                    
                    bbox = np.zeros(4, dtype=np.float32)
                    bbox[0]=float(bndbox.find('xmin').text)
                    bbox[1]=float(bndbox.find('ymin').text)
                    bbox[2]=float(bndbox.find('xmax').text)
                    bbox[3]=float(bndbox.find('ymax').text)
                    
                    bbox*=scale
                    bbox[[0,2]]+=xstart
                    bbox[[1,3]]+=ystart
    
                    if ind==inds[0]:
                        ref_boxes[obj_id]=bbox
                    else:
                        det_boxes[obj_id]=bbox

                if ind==inds[0] and len(ref_boxes.keys())>0:
                    temp_image=image[np.newaxis, :,:,:].astype(np.float32)
                    non_empty_batch=True
                elif ind==inds[1] and len(ref_boxes.keys())>0:
                    det_image=image[np.newaxis, :,:,:].astype(np.float32)
                
            ref_boxes_align, det_boxes_align=self.align_boxes(ref_boxes, det_boxes)
            
            if len(ref_boxes_align)>0:
                roidb['raw_temp_boxes']=ref_boxes_align
                temp_boxes=util.compute_template_boxes(ref_boxes_align, (temp_image.shape[2], temp_image.shape[1]), gain=self.margin_gain, shape='same')
                roidb['temp_image']=temp_image
                roidb['det_image']=det_image
                roidb['temp_boxes']=temp_boxes
                roidb['det_boxes']=det_boxes_align
                
                '''NHWC'''
                bound=(det_image.shape[2], det_image.shape[1])
                search_boxes=util.calc_search_boxes(temp_boxes, bound)
                roidb['search_boxes']=search_boxes
                roidb['bound']=bound
                '''
                Next
                roidb['bbox_overlaps']
                '''
#                anchors=self.gen_anchors(search_boxes, bound)
                anchors=G.gen_region_anchors(self.raw_anchors, search_boxes, bound, stride=self.stride, K=self.K, rpn_conv_size=self.roi_size)
                bbox_overlaps=U.bbox_overlaps_per_image(np.vstack(anchors), det_boxes_align)
                roidb['anchors']=anchors
                roidb['bbox_overlaps']=bbox_overlaps
                roidbs.append(roidb)
        '''
        [NHWC,NHWC]
        '''
#        assert len(ref_images)==len(box_sizes), 'Images and size array must have the same length: {} v.s. {}'.format(len(ref_images), len(box_sizes))
        
        for ind, v in tmp.items():
            self.index_per_seq[ind]+=v
        
        index_res=self.index_per_seq-self.upper_bound_per_seq
        index_res=index_res[self.permute_inds[[valid_seq_inds]]]
        invalid_seq_inds=np.where(index_res>0)[0]
        
        num_sequences=num_valid_seqs-len(invalid_seq_inds)
               
        if num_sequences==0:
            self.index_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
            self.round_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
            self.inds = np.random.permutation(np.arange(self.num_sequences))
            self.index=0
            self.iter_stop=True
        else:    
            self.index+=self.batch_size
            if self.index>=num_sequences:
                self.index=0
            else:
                subtract_inds=np.where(invalid_seq_inds<=self.index)[0]                
                self.index-=len(subtract_inds)
            
        return roidbs, non_empty_batch