from .dataset import Dataset
import roidb.image_utils as util
import os
import os.path as op
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from core.config import cfg

class Detrac(Dataset):
    def __init__(self, im_width=0, im_height=0, name='Detrac', load_gt=True):
        super(Detrac, self).__init__(im_width=im_width, im_height=im_height, name=name, load_gt=load_gt)

        print('Using benchmark {}'.format(self.dataset_name))
        self.data_dir = op.join(cfg.DATA_DIR, 'Insight-MVT_Annotation_Train')
        if self.load_gt:
            self.anno_dir = op.join(cfg.DATA_DIR, 'DETRAC-Train-Annotations-XML')        

        self.get_dataset()
        assert len(self.dataset)==len(self.annotations), 'Dataset and annotations not uniformed'
        self.index=0

    def get_dataset(self):
        seqs=sorted(os.listdir(self.data_dir)) 
        if self.load_gt:       
            anno_files=sorted(os.listdir(self.anno_dir))
        detrac_all_dirs={}
        seq_ind_map={}

        self.num_sequences=0
        for i in range(len(seqs)):
            seq=seqs[i]
            seq_ind_map[seq]=i
            detrac_all_dirs[i]=seq
            self.num_sequences+=1

        self.seq_ind_map=seq_ind_map
        self.dataset=detrac_all_dirs
        if self.load_gt:
            self.annotations=anno_files

    def choice(self, seq_name=None):
        if seq_name is None:
            self.choice_img_dir=op.join(self.data_dir,self.dataset[0])
            if self.load_gt:
                self.choice_anno_file=op.join(self.anno_dir,self.annotations[0]) 
        else:
            assert seq_name in self.seq_ind_map.keys(), '{} not exists'.format(seq_name)
            seq_ind=self.seq_ind_map[seq_name]
            self.choice_img_dir=op.join(self.data_dir, self.dataset[seq_ind])
            if self.load_gt:
                self.choice_anno_file=op.join(self.anno_dir, self.annotations[seq_ind])    

        self.image_files=sorted(os.listdir(self.choice_img_dir))
        if self.load_gt:
            self.gt_boxes=self.get_gt_boxes(self.choice_anno_file)
        self.num_samples=len(self.image_files)
        self.index=0

    def get_gt_boxes(self, anno_file):
        gt_boxes=[]
        tree = ET.parse(anno_file)
        root = tree.getroot()

        frames=root.findall('frame')
        for i, frame in enumerate(frames):
            target_list=frame.find('target_list')
            targets=target_list.findall('target')
            if len(targets) == 0:
                print('{} has no targets'.format(op.join(self.choice_anno_dir, self.image_files[i])))
                gt_boxes.append([])
                continue

            gt_boxes_this_image=np.zeros((0, 4), dtype=np.int32)
            targets=sorted(targets, key=lambda x:int(x.attrib['id']))
            
            for obj in targets:
                bbox = np.zeros(4, dtype=np.float32)
                bbox_attribs=obj.find('box').attrib

                left=float(bbox_attribs['left'])
                top=float(bbox_attribs['top'])
                width=float(bbox_attribs['width'])
                height=float(bbox_attribs['height'])

                bbox[0]=left
                bbox[1]=top
                bbox[2]=left+width-1
                bbox[3]=top+height-1
                gt_boxes_this_image=np.append(gt_boxes_this_image, bbox.reshape(1,4), 0)
            gt_boxes.append(gt_boxes_this_image)
        return gt_boxes

    def __len__(self):
        return self.num_sequences

    def __getitem__(self):
        ind=self.index
        if ind<self.num_samples:
            image_file=op.join(self.choice_img_dir, self.image_files[ind])
            gt_boxes=self.gt_boxes[ind]
            image=cv2.imread(image_file)         
            if self.load_gt:   
                image_scaled, gt_boxes=self.imresize(image, gt_boxes)  
                self.index+=1            
                return image_scaled, gt_boxes
            else:
                image_scaled=cv2.resize(image, (self.im_w, self.im_h), interpolation=cv2.INTER_LINEAR)
                self.index+=1            
                return image_scaled
        else:
            return None