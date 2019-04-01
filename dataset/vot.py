from .dataset import Dataset
import os
import os.path as op
import numpy as np
import cv2
from core.config import cfg
import rpn.generate_anchors as G

VOT_DIR='/mnt/sda7/vot2016_part'

class VOT(Dataset):
    def __init__(self, im_width=0, im_height=0, name='VOT'):
        super(VOT, self).__init__(im_width=im_width, im_height=im_height, name=name)
        print('Using benchmark {}'.format(self.dataset_name))
        self.get_dataset()
        assert len(self.dataset)==len(self.annotations), 'Dataset and annotations not uniformed'

        self.index=0
        self.choice()
        
    def get_dataset(self):
        vot_sub_dirs=os.listdir(VOT_DIR)
        vot_all_dirs={}
        vot_all_annotations={}
        seq_ind_map={}

        self.num_sequences=0
        for i, sub_dir in enumerate(vot_sub_dirs):
            img_dir=sub_dir
            anno_file=op.join(sub_dir, 'groundtruth.txt')
            seq_ind_map[sub_dir]=i
            vot_all_dirs[i]=img_dir
            vot_all_annotations[i]=anno_file
            self.num_sequences+=1

        self.seq_ind_map=seq_ind_map
        self.dataset=vot_all_dirs
        self.annotations=vot_all_annotations

    def _permute(self):
        self.inds=np.random.permutation(np.arange(self.num_sequences))
        self.index=0

    def choice(self, seq_name=None):
        if seq_name is None:
            self.choice_img_dir=op.join(VOT_DIR,self.dataset[0])
            self.choice_annotation=op.join(VOT_DIR,self.annotations[0]) 
        else:
            assert seq_name in self.seq_ind_map.keys(), '{} not exists'.format(seq_name)
            seq_ind=self.seq_ind_map[seq_name]
            self.choice_img_dir=op.join(VOT_DIR, self.dataset[seq_ind])
            self.choice_annotation=op.join(VOT_DIR, self.annotations[seq_ind])    
        with open(self.choice_annotation, 'r') as f:
            lines=f.readlines()
        img_files=os.listdir(self.choice_img_dir)
        img_files=[f for f in img_files if '.jpg' in f]
        self.image_files=sorted(img_files)
        self.gt_boxes=self.get_gt_boxes(lines)
        self.num_samples=len(self.image_files)
        self.index=0

    def get_gt_boxes(self, lines):
        gt_boxes=[]
        for line in lines:
            if len(line)<=1:
                continue
            line=line.rstrip()
            split=' '
            if ',' in line:
                split=','
            elif '\t' in line:
                split='\t'
            items=line.split(split)
#            print(items)
            gt_line=list(map(float, items))
            assert len(gt_line)==8 or len(gt_line)==4, 'Invalid groundtruth'
            bbox=np.asarray(gt_line, dtype=np.float32)
            xs=bbox[::2]
            ys=bbox[1::2]
            xmin=np.min(xs)
            xmax=np.max(xs)
            ymin=np.min(ys)
            ymax=np.max(ys)
            bbox=np.asarray([[xmin,ymin,xmax,ymax]])
            gt_boxes.append(bbox)
        return np.vstack(gt_boxes)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self):
        ind=self.index
        if ind<self.num_samples:
            image_file=op.join(self.choice_img_dir, self.image_files[ind])
#            print(image_file)
            gt_boxes=self.gt_boxes[ind].reshape(-1,4)
            image=cv2.imread(image_file)
            image_scaled, gt_boxes_scaled=self.imresize(image, gt_boxes)
            self.index+=1
            return image_scaled, gt_boxes_scaled
        else:
            return None