import os.path as op
import numpy as np
import cv2
from model.mot_forward import MotFRCNN
from core.config import cfg
from dataset import detrac,vid,vot
from dataset.data_loader import DataLoader
from inference import CLASSES, inference_detect
import roidb.box_utils as butil
import rpn.generate_anchors as G
import sys
import os
import os.path as op

im_width=640
im_height=384

K=len(cfg.RATIOS)*len(cfg.SCALES)
bound=(im_width, im_height)
out_size=(im_width//8, im_height//8)
det_raw_anchors=G.generate_anchors(cfg.BASIC_SIZE, cfg.RATIOS, cfg.SCALES)
dummy_search_box=np.array([[0,0,im_width-1,im_height-1]])
det_anchors=G.gen_region_anchors(det_raw_anchors, dummy_search_box, bound, K=K, size=out_size)[0]
bound=(im_width, im_height)

def compute_progress(max_val, length):  
    progress=np.linspace(0,max_val-1, length).astype(np.int32).tolist()
    steps=[]
    
    if length<=max_val:
        diff=[progress[i]-progress[i-1] for i in range(1,len(progress))]
        for d in diff:
            steps.append(1)
            for _ in range(d-1):
                steps.append(0)  
        steps.append(1)              
    else:
        ind=0
        while ind<length:
            cnt=progress.count(progress[ind])
            steps.append(cnt)
            ind+=cnt
    return steps

def main(dataset_obj, seqName, model=None):    
#    video_path='./result.avi'
#    writer=cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (im_width, im_height))
    dataset_obj.choice(seqName)
    loader=DataLoader(dataset_obj)

    roidb={'anchors':det_anchors, 'bound':bound}

    num_frames=dataset_obj.__len__()
    print('{} has {} samples'.format(seqName, num_frames))

    progress_len=80
    steps=compute_progress(num_frames, progress_len)
    with open('./det_results/{}_Det_DLMOT.txt'.format(seqName), 'w') as f:
        for i in range(progress_len):
            print('=',end='')
        print('\r',end='')

        pair_images=[]
        for index, image in enumerate(loader):
            for i in range(steps[index]):
                print('>',end='', flush=True)

            pair_images.append(image)
            if len(pair_images)==1:
                continue

            roidb['temp_image']=pair_images[0][np.newaxis,:,:,:].astype(np.float32, copy=True)
            roidb['det_image']=pair_images[1][np.newaxis,:,:,:].astype(np.float32, copy=True)

            outputs=inference_detect(model, roidb, batch_size=2)

            fx=1.0*960/im_width
            fy=1.0*540/im_height

            for ix, output in enumerate(outputs):
                cls_boxes=output['cls_bboxes']
                frame_idx=index+ix
                obj_index=1

                for i in range(1, len(CLASSES)):
                    bboxes=cls_boxes[i]
                    for bbox in bboxes:
                        bbox[[0,2]]*=fx
                        bbox[[1,3]]*=fy
                        x,y,w,h,c=bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1],bbox[4]
                        
                        line='{},{},{},{},{},{},{}'.format(frame_idx, obj_index,x,y,w,h,c)
                        obj_index+=1

                        f.write(line+'\n')
            pair_images.clear()

def main2(dataset_obj, seqName, model=None):
    loader=DataLoader(dataset_obj)
#    video_path='./result.avi'
#    writer=cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (im_width, im_height))
    dataset_obj.choice(seqName)
    roidb={'anchors':det_anchors, 'bound':bound}

    num_frames=dataset_obj.__len__()
    print('{} has {} samples'.format(seqName, num_frames))

    progress_len=80
    steps=compute_progress(num_frames, progress_len)
    with open('./det_results/{}_Det_DLMOT.txt'.format(seqName), 'w') as f:
        for i in range(progress_len):
            print('=',end='')
        print('\r',end='')
        for index, image in enumerate(loader):
            for i in range(steps[index]):
                print('>',end='',flush=True)

            roidb['temp_image']=image[np.newaxis,:,:,:].astype(np.float32, copy=True)
            roidb['det_image']=image[np.newaxis,:,:,:].astype(np.float32, copy=True)

            outputs=inference_detect(model, roidb, batch_size=1)

            output=outputs[0]
            cls_boxes=output['cls_bboxes']

            obj_index=1
            fx=1.0*960/image.shape[1]
            fy=1.0*540/image.shape[0]

            for i in range(1, len(CLASSES)):
                bboxes=cls_boxes[i]
                for bbox in bboxes:
                    bbox[[0,2]]*=fx
                    bbox[[1,3]]*=fy
                    x,y,w,h,c=bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1],bbox[4]
                    line='{},{},{},{},{},{},{}'.format(index+1, obj_index,x,y,w,h,c)
                    obj_index+=1

                    f.write(line+'\n')

if __name__=='__main__':
    model_path='./ckpt/dl_mot_epoch_40.pkl'
#    model_path='./ckpt/dl_mot_iter_660000.pkl'
    cfg.PHASE='TEST'
    cfg.TEST.RPN_PRE_NMS_TOP_N=8000
    cfg.TEST.RPN_POST_NMS_TOP_N=2000
    cfg.DET_SCORE_THRESH=0.5
    cfg.TEST.RPN_NMS_THRESH=0.7
    cfg.TEST.NMS_THRESH=0.7
    cfg.TEST.DATASET='detrac'
    dataset_obj=detrac.Detrac(im_width=im_width, im_height=im_height, name='Detrac', load_gt=False)
#    dataset_obj.choice('MVI_40201')
    model=MotFRCNN(im_width, im_height, pretrained=False)
    model.load_weights(model_path)
    model.cuda()
    
    test_data_dir='/mnt/sda7/DETRAC-train-data/Insight-MVT_Annotation_Small'
    seqs=os.listdir(test_data_dir)
    print('Sequences for testing: ')
    print(seqs)

    for ix, seq in enumerate(seqs):
        print('Processing {}/{}:{}'.format(ix+1, len(seqs), seq))
        main(dataset_obj, seq, model=model)
        print('\n')