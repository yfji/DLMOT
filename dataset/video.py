from .dataset import Dataset
import roidb.image_utils as util
import os
import os.path as op
import numpy as np
import cv2

class Video(Dataset):
    def __init__(self, im_width=0, im_height=0, name='VIDEO'):
        super(Video, self).__init__(im_width=im_width, im_height=im_height, name=name)
        print('Using benchmark {}'.format(self.dataset_name))
        self.data_dir = '/home/yfji/Workspace/PyTorch/DLMOT_Final3/videos'
        self.get_dataset()
        self.index=0
        
    def get_dataset(self):
        videos=sorted(os.listdir(self.data_dir))
        all_videos={}
        seq_ind_map={}

        self.num_sequences=0
        for i in range(len(videos)):
            video=videos[i]
            seq_ind_map[video]=i
            all_videos[i]=video
            self.num_sequences+=1

        self.seq_ind_map=seq_ind_map
        self.dataset=all_videos

    def choice(self, seq_name=None):
        if seq_name is None:
            self.choice_video=op.join(self.data_dir,self.dataset[0])
        else:
            assert seq_name in self.seq_ind_map.keys(), '{} not exists'.format(seq_name)
            seq_ind=self.seq_ind_map[seq_name]
            self.choice_video=op.join(self.data_dir, self.dataset[seq_ind])

        self.capture=cv2.VideoCapture(self.choice_video)
        self.num_samples=self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.index=0

    def __len__(self):
        return self.num_sequences

    def __getitem__(self):
        ind=self.index
        if ind<self.num_samples:
            ok,image=self.capture.read()
            if not ok:
                return None         
            image_scaled=cv2.resize(image, (self.im_w, self.im_h), interpolation=cv2.INTER_LINEAR)

            self.index+=1            
            return image_scaled
        else:
            return None

if __name__=='__main__':
    video_dataset=Video(640, 384)
    video_dataset.choice('beisanhuan.mp4')
    while True:
        image=video_dataset.__getitem__()
        if image is None:
            break
        else:
            cv2.imshow('video', image)
            if cv2.waitKey(1)==27:
                break