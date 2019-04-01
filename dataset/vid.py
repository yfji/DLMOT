from .dataset import Dataset
import roidb.image_utils as util
import os
import os.path as op
import numpy as np
import cv2
import xml.etree.ElementTree as ET

#VID_IMG_DIR='/mnt/sda7/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000'
VID_IMG_DIR='/mnt/sda7/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0001'
VID_ANNO_DIR='/mnt/sda7/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0001'

class VID(Dataset):
    def __init__(self, im_width=0, im_height=0, name='VID'):
        super(VID, self).__init__(im_width=im_width, im_height=im_height, name=name)
        print('Using benchmark {}'.format(self.dataset_name))
        self.get_dataset()
        assert len(self.dataset)==len(self.annotations), 'Dataset and annotations not uniformed'
        self.index=0

    def get_dataset(self):
        vid_sub_dirs=sorted(os.listdir(VID_ANNO_DIR))
        anno_sub_dirs=sorted(os.listdir(VID_ANNO_DIR))
        vid_all_dirs={}
        vid_all_annotations={}
        seq_ind_map={}

        self.num_sequences=0
        for i in range(len(vid_sub_dirs)):
            sub_dir=vid_sub_dirs[i]
            anno_dir=anno_sub_dirs[i]
            seq_ind_map[sub_dir]=i
            vid_all_dirs[i]=sub_dir
            vid_all_annotations[i]=anno_dir
            self.num_sequences+=1

        self.seq_ind_map=seq_ind_map
        self.dataset=vid_all_dirs
        self.annotations=vid_all_annotations

    def choice(self, seq_name=None):
        if seq_name is None:
            self.choice_img_dir=op.join(VID_IMG_DIR,self.dataset[0])
            self.choice_anno_dir=op.join(VID_IMG_DIR,self.annotations[0]) 
        else:
            assert seq_name in self.seq_ind_map.keys(), '{} not exists'.format(seq_name)
            seq_ind=self.seq_ind_map[seq_name]
            self.choice_img_dir=op.join(VID_IMG_DIR, self.dataset[seq_ind])
            self.choice_anno_dir=op.join(VID_ANNO_DIR, self.annotations[seq_ind])    

        self.image_files=sorted(os.listdir(self.choice_img_dir))
        self.gt_boxes=self.get_gt_boxes(self.choice_anno_dir)
        self.num_samples=len(self.image_files)
        self.index=0

    def get_gt_boxes(self, anno_dir):
        gt_boxes=[]
        anno_files=sorted(os.listdir(anno_dir))
        for i, anno_file in enumerate(anno_files):
            xml_file=op.join(anno_dir, anno_file)
            tree = ET.parse(xml_file)
            root = tree.getroot()

            objects = root.findall('object')
            if len(objects) == 0:
                print('{} has no targets'.format(op.join(self.choice_anno_dir, self.image_files[i])))
                gt_boxes.append([])
                continue

            gt_boxes_this_image=np.zeros((0, 4), dtype=np.int32)
            objects=sorted(objects, key=lambda x:int(x.find('trackid').text))
            
            for obj in objects:
                bndbox = obj.find('bndbox')

                bbox = np.zeros(4, dtype=np.float32)
                bbox[0]=float(bndbox.find('xmin').text)
                bbox[1]=float(bndbox.find('ymin').text)
                bbox[2]=float(bndbox.find('xmax').text)
                bbox[3]=float(bndbox.find('ymax').text)

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
            image_scaled, gt_boxes=self.imresize(image, gt_boxes)
            '''
            image_scaled, (xstart,ystart), scale=util.resize_and_pad_image(image, self.im_w, self.im_h)
            gt_boxes*=scale
            gt_boxes[:,[0,2]]+=xstart
            gt_boxes[:,[1,3]]+=ystart
            '''
            self.index+=1            
            return image_scaled, gt_boxes
        else:
            return None
