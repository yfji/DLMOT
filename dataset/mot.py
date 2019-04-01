from .dataset import Dataset
import os
import os.path as op
import cv2
import numpy as np

MOT_SUBDIRS = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']

class MOT(Dataset):
    def __init__(self, im_width=0, im_height=0, name='MOT'):
        super(MOT, self).__init__(im_width=im_width, im_height=im_height, name=name)
        self.mot_dir='/mnt/sda7/MOT2017/train'
        self.get_dataset()
        '''bounding box rect in first frame'''
        self.first_rect=None
        self.person_conf = 0.6
        
    def get_dataset(self):
        mot_all_dirs={}
        seq_ind_map={}
        self.num_sequences=0
        for i, sub_dir in enumerate(MOT_SUBDIRS):
            seq_ind_map[sub_dir]=i
            mot_all_dirs[i]=sub_dir
            self.num_sequences+=1
        self.seq_ind_map=seq_ind_map
        self.dataset=mot_all_dirs

    def wh2xy(self, bbox):
        return np.asarray([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
      
    def get_gt_boxes(self, anno_file):
        with open(anno_file, 'r') as f:
                lines = f.readlines()
        gt_boxes=[]
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
        for idx, sample in sample_subdir.items():
            gt_box=np.zeros((0,4),dtype=np.float32)
            sort_samples=sorted(sample, key=lambda x:x['id'])
            for item in sort_samples:
                gt_box=np.append(gt_box, item['bbox'].reshape(1,-1), 0)
            gt_boxes.append(gt_box)
        return gt_boxes

    def get_first_rect(self):
        return None
    
    def choice(self, seq_name=None):
        if seq_name is None:
            self.choice_img_dir=op.join(self.mot_dir, self.dataset[0])
        else:
            assert seq_name in self.seq_ind_map.keys(), '{} not exists'.format(seq_name)
            seq_ind=self.seq_ind_map[seq_name]
            self.choice_img_dir=op.join(self.mot_dir, self.dataset[seq_ind], 'img1')
            self.choice_anno_file=op.join(self.mot_dir, self.dataset[seq_ind], 'gt', 'gt.txt')
        self.image_files=sorted(os.listdir(self.choice_img_dir))
        self.gt_boxes=self.get_gt_boxes(self.choice_anno_file)
        self.num_samples=len(self.image_files)
        self.index=0
        
    def __len__(self):
        return self.num_sequences

    def __getitem__(self):
        ind=self.index
        if ind<self.num_samples:
            image_file=op.join(self.choice_img_dir, self.image_files[ind])
            image=cv2.imread(image_file)
#            gt_box=self.first_rect if self.index==0 else None
            gt_box=self.gt_boxes[ind]
            image_scaled, gt_boxes_scaled=self.imresize(image, gt_box)
            self.index+=1
            return image_scaled, gt_boxes_scaled
#            return image_scaled
        else:
            return None