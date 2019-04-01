from .dataset import Dataset
import roidb.image_utils as util
import os
import os.path as op
import numpy as np
import cv2

class KITTI(Dataset):
    def __init__(self, im_width=0, im_height=0, name='KITTI'):
        super(KITTI, self).__init__(im_width=im_width, im_height=im_height, name=name)
        print('Using benchmark {}'.format(self.dataset_name))
        self.data_dir = '/mnt/sda7/KITTI/data_tracking_image_2/testing/image_02'
        self.get_dataset()
        self.index=0

    def get_dataset(self):
        seqs=sorted(os.listdir(self.data_dir))
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

    def choice(self, seq_name=None):
        if seq_name is None:
            self.choice_img_dir=op.join(self.data_dir,self.dataset[0])
        else:
            assert seq_name in self.seq_ind_map.keys(), '{} not exists'.format(seq_name)
            seq_ind=self.seq_ind_map[seq_name]
            self.choice_img_dir=op.join(self.data_dir, self.dataset[seq_ind])

        self.image_files=sorted(os.listdir(self.choice_img_dir))
        self.num_samples=len(self.image_files)
        self.index=0

    def __len__(self):
        return self.num_sequences

    def __getitem__(self):
        ind=self.index
        if ind<self.num_samples:
            image_file=op.join(self.choice_img_dir, self.image_files[ind])
            image=cv2.imread(image_file)            
            image_scaled=cv2.resize(image, (self.im_w, self.im_h), interpolation=cv2.INTER_LINEAR)

            self.index+=1            
            return image_scaled
        else:
            return None

