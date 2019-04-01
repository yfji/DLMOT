import numpy as np
from rpn.util import bbox_transform
from rpn.util import bbox_overlaps_batch, bbox_overlaps_per_image
from core.config import cfg

'''
anchors_with_shift
[anchors_box1, anchors_box2,...,anchors_box_n]
aranged by the search_boxes of last frame

gt_boxes
[gt_box1, gt_box2,...,gt_box_n]
aranged by the gt_boxes of current frame
'''

DEBUG=False
#def compute_track_rpn_targets(anchors, gt_box, search_box, det_box, bbox_overlaps=None, bound=None, rpn_conv_size=10, K=9):
def compute_track_rpn_targets(track_anchors, det_boxes, bbox_overlaps=None, bound=None, rpn_conv_size=10, K=6, num_boxes=None):
    anchor_cls_targets=np.zeros((0, K*rpn_conv_size, rpn_conv_size), dtype=np.int32)
    anchor_bbox_targets=np.zeros((0, 4*K, rpn_conv_size, rpn_conv_size),dtype=np.float32)
    bbox_weights=np.zeros((0, 4*K, rpn_conv_size, rpn_conv_size), dtype=np.float32)
    
    if bbox_overlaps is None:
        bbox_overlaps=bbox_overlaps_batch(track_anchors, det_boxes, num_boxes, branch='rpn')
    
    fg_anchor_inds=[]
    bg_anchor_inds=[]

    start=0
    for i, box_size in enumerate(num_boxes):
        left=start
        right=left+box_size  

        anchors=np.vstack(track_anchors[left:right])  #all anchors per ref box!!!!!
        bbox_overlap=np.asarray(bbox_overlaps[i])
        max_overlap=np.max(bbox_overlap, axis=1)

        '''cls start'''
        anchor_cls_target=np.zeros(anchors.shape[0], dtype=np.int32)-100    #box_size*A*K-->N*A*K
        fg_anchor_ind=np.where(max_overlap>=cfg[cfg.PHASE].TRACK_RPN_POSITIVE_THRESH)[0]
        bg_anchor_ind=np.where(max_overlap<cfg[cfg.PHASE].TRACK_RPN_NEGATIVE_THRESH)[0]
        
        fg_anchor_inds.append(fg_anchor_ind)
        bg_anchor_inds.append(bg_anchor_ind)

        anchor_cls_target[fg_anchor_ind]=1
        anchor_cls_target[bg_anchor_ind]=0

        anchor_cls_target=anchor_cls_target.reshape(box_size, rpn_conv_size, rpn_conv_size, K).\
            transpose(0,3,2,1).reshape(box_size, K*rpn_conv_size, rpn_conv_size)
        anchor_cls_targets=np.append(anchor_cls_targets, anchor_cls_target, 0)
        '''cls end'''
        '''bbox start'''
        bbox_loss_inds=fg_anchor_ind
        mask_inds=np.zeros((anchors.shape[0], 4), dtype=np.float32)
        mask_inds[bbox_loss_inds,:]=1
        gt_boxes_sample=det_boxes[left:right]

        gt_rois=gt_boxes_sample[np.argmax(bbox_overlap, axis=1)]

        bbox_deltas=bbox_transform(anchors, gt_rois)
        if cfg.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            bbox_deltas=bbox_deltas/cfg.RPN_BBOX_STD_DEV
            
        bbox_deltas=bbox_deltas.reshape(box_size,rpn_conv_size, rpn_conv_size, 4*K).transpose((0,3,1,2))
        anchor_bbox_targets=np.append(anchor_bbox_targets, bbox_deltas, 0)
        '''bbox end'''
        '''bbox weights'''
        mask_inds=mask_inds.reshape(box_size, rpn_conv_size, rpn_conv_size, 4*K).transpose(0,3,1,2)
        bbox_weights=np.append(bbox_weights, mask_inds, 0)
        start+=box_size

    return anchor_cls_targets, anchor_bbox_targets, bbox_weights, fg_anchor_inds
    
def compute_detection_rpn_targets(anchors, gt_boxes, out_size, bbox_overlaps=None, K=9, batch_size=2): 
    anchor_cls_targets=np.zeros((0, K*out_size[1], out_size[0]), dtype=np.int32)
    anchor_bbox_targets=np.zeros((0, 4*K, out_size[1], out_size[0]),dtype=np.float32)
    fg_anchor_inds=[]
    bg_anchor_inds=[]

    if bbox_overlaps is None:
        bbox_overlaps=[]
        for i in range(batch_size):
            bbox_overlaps.append(bbox_overlaps_per_image(anchors, gt_boxes[i], branch='frcnn'))
        
    bbox_weights=np.zeros((0,4*K, out_size[1], out_size[0]), dtype=np.float32)
    
    for i in range(batch_size):
        bbox_overlap=np.asarray(bbox_overlaps[i])
        max_overlap=np.max(bbox_overlap, axis=1)

        '''cls start'''
        anchor_cls_target=np.zeros(anchors.shape[0], dtype=np.int32)-100    #box_size*A*K-->N*A*K
        fg_anchor_ind=np.where(max_overlap>=cfg[cfg.PHASE].RPN_POSITIVE_THRESH)[0]
        bg_anchor_ind=np.where(max_overlap<cfg[cfg.PHASE].RPN_NEGATIVE_THRESH)[0]

        fg_anchor_inds.append(fg_anchor_ind)
        bg_anchor_inds.append(bg_anchor_ind)

        anchor_cls_target[fg_anchor_ind]=1
        anchor_cls_target[bg_anchor_ind]=0
        
        anchor_cls_target=anchor_cls_target.reshape(1, out_size[1], out_size[0], K).\
            transpose(0,3,2,1).reshape(1, K*out_size[1], out_size[0])
        anchor_cls_targets=np.append(anchor_cls_targets, anchor_cls_target, 0)
        '''cls end'''
        
        '''bbox start'''
        bbox_loss_inds=fg_anchor_ind
        mask_inds=np.zeros((anchors.shape[0], 4), dtype=np.float32)
        mask_inds[bbox_loss_inds,:]=1

        gt_boxes_this_image=gt_boxes[i]
        
        gt_rois=gt_boxes_this_image[np.argmax(bbox_overlap, axis=1)]
        bbox_deltas=bbox_transform(anchors, gt_rois)
        
        if cfg.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            bbox_deltas=bbox_deltas/cfg.RPN_BBOX_STD_DEV
            
#        bbox_deltas*=mask_inds  
        bbox_deltas=bbox_deltas.reshape(1,out_size[1], out_size[0], 4*K).transpose((0,3,1,2))
        anchor_bbox_targets=np.append(anchor_bbox_targets, bbox_deltas, 0)
        '''bbox end'''

        '''bbox weights'''
        mask_inds=mask_inds.reshape(1, out_size[1], out_size[0], 4*K).transpose(0,3,1,2)
        bbox_weights=np.append(bbox_weights, mask_inds, 0)

    return anchor_cls_targets, anchor_bbox_targets, bbox_weights, fg_anchor_inds


if __name__=='__main__':
    im_w=640
    im_h=384
    stride=8
    rpn_conv_size=14-5+1

    DEBUG=True

    bound=(im_w, im_h)
    out_size=(im_w//stride, im_h//stride)

    basic_size=cfg.BASIC_SIZE
    scales=cfg.SCALES
    ratios=cfg.RATIOS
    cfg.PHASE='TRAIN'

    K=len(scales)*len(ratios)
    import rpn.generate_anchors as G

    raw_anchors=G.generate_anchors(basic_size, ratios, scales)
    dummy_search_box=np.asarray([0,0,im_w-1,im_h-1]).reshape(1,-1)

    anchors=G.gen_region_anchors(raw_anchors, dummy_search_box, bound, K, out_size)[0]
    print(anchors.shape)
    gt_box=np.array([20,20,140,100])
    search_box=np.array([0,0,200,140])
    det_box=np.array([30,25,156,114])

    fg_anchors_global, fg_anchors_local, fg_anchors_ind_local, xs, ys=compute_track_rpn_targets(anchors, gt_box, search_box, det_box, None, rpn_conv_size=rpn_conv_size, K=K)

    import cv2
    canvas=np.zeros((im_h,im_w,3), dtype=np.uint8)+255

    cv2.rectangle(canvas, (search_box[0],search_box[1]),(search_box[2], search_box[3]), (0,0,0), 2)
    cv2.rectangle(canvas, (det_box[0],det_box[1]),(det_box[2], det_box[3]), (0,0,255), 1)

    sx,sy=search_box[0:2]
    sw=search_box[2]-sx+1
    sh=search_box[3]-sy+1

    xdiff=1.0*sw/rpn_conv_size
    ydiff=1.0*sh/rpn_conv_size

    for i in range(rpn_conv_size):
        cv2.line(canvas, (sx+int(i*xdiff), sy),(sx+int(i*xdiff), sy+sh-1), (0,0,0), 1)
    for i in range(rpn_conv_size):
        cv2.line(canvas, (sx, sy+int(i*ydiff)),(sx+sw-1, sy+int(i*ydiff)), (0,0,0), 1)

    print(len(fg_anchors_global))
    x1,y1,x2,y2=np.split(fg_anchors_global, 4, axis=1)
    ctrxs=0.5*(x1+x2)
    ctrys=0.5*(y1+y2)

    for anchor in fg_anchors_global:
        anchor=anchor.astype(np.int32)
        cv2.rectangle(canvas, (anchor[0],anchor[1]),(anchor[2],anchor[3]), (0,255,0),1)
    for i in range(len(fg_anchors_global)):
        cx=int(ctrxs[i])
        cy=int(ctrys[i])
        cv2.circle(canvas, (cx,cy), 2, (255,0,0), -1)

    xys=np.meshgrid(xs,ys)
    xs=xys[0].ravel()
    ys=xys[1].ravel()
    for x,y in zip(xs,ys):
        cv2.circle(canvas, (int(x),int(y)), 2, (180,180,180), -1)

    for i in fg_anchors_ind_local:
        cv2.circle(canvas, (int(xs[i]),int(ys[i])), 2, (0,0,255), -1)

    print(fg_anchors_global.shape)
    print(fg_anchors_local.shape)

    for anchor in fg_anchors_local:
        anchor=anchor.astype(np.int32)
        cv2.rectangle(canvas, (anchor[0],anchor[1]),(anchor[2],anchor[3]), (0,0,255),1)
    cv2.imshow('anchors', canvas)
    cv2.waitKey()