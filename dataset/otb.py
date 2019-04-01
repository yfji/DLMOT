from .dataset import Dataset
import os
import os.path as op
import numpy as np
import cv2

OTB_DIR='/mnt/sda7/OTB'

class OTB(Dataset):
    def __init__(self, im_width=0, im_height=0, name='OTB'):
        super(OTB, self).__init__(im_width=im_width, im_height=im_height, name=name)
        print('Using benchmark {}'.format(self.dataset_name))
        self.get_dataset()
        assert len(self.dataset)==len(self.annotations), 'Dataset and annotations not uniformed'

        self.index=0
        self.choice()
        
    def get_dataset(self):
        otb_sub_dirs=os.listdir(OTB_DIR)
        otb_all_dirs={}
        otb_all_annotations={}
        seq_ind_map={}

        self.num_sequences=0
        for i, sub_dir in enumerate(otb_sub_dirs):
            img_dir=op.join(sub_dir, 'img')
            anno_file=op.join(sub_dir, 'groundtruth_rect.txt')
            seq_ind_map[sub_dir]=i
            otb_all_dirs[i]=img_dir
            otb_all_annotations[i]=anno_file
            self.num_sequences+=1

        self.seq_ind_map=seq_ind_map
        self.dataset=otb_all_dirs
        self.annotations=otb_all_annotations

    def _permute(self):
        self.inds=np.random.permutation(np.arange(self.num_sequences))
        self.index=0

    def choice(self, seq_name=None):
        if seq_name is None:
            self.choice_img_dir=op.join(OTB_DIR,self.dataset[0])
            self.choice_annotation=op.join(OTB_DIR,self.annotations[0]) 
        else:
            assert seq_name in self.seq_ind_map.keys(), '{} not exists'.format(seq_name)
            seq_ind=self.seq_ind_map[seq_name]
            self.choice_img_dir=op.join(OTB_DIR, self.dataset[seq_ind])
            self.choice_annotation=op.join(OTB_DIR, self.annotations[seq_ind])    
        with open(self.choice_annotation, 'r') as f:
            lines=f.readlines()
        self.image_files=sorted(os.listdir(self.choice_img_dir))
        self.gt_boxes=self.get_gt_boxes(lines)
        self.num_samples=len(self.image_files)
        self.index=0

    def get_gt_boxes(self, lines):
        gt_boxes=[]
        for line in lines:
            if len(line)==0:
                continue
            line=line.rstrip()
            split=' '
            if ',' in line:
                split=','
            elif '\t' in line:
                split='\t'
            items=line.split(split)
            box=np.asarray(list(map(int, items)), dtype=np.float32)
            box[[2,3]]+=box[[0,1]]
            gt_boxes.append(box)
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