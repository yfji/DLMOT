import numpy as np
from rpn.util import bbox_overlaps_per_image
from rpn.util import bbox_transform
from core.config import cfg

def get_proposal_target(proposal_batch, gt_boxes, gt_classes, batch_size):
    rois_per_image=cfg[cfg.PHASE].BATCH_SIZE
    fg_rois_per_image=int(rois_per_image*cfg[cfg.PHASE].FG_FRACTION)
    
    num_fg_rois=0
    num_bg_rois=0
    
    proposal_cls_targets=np.zeros(0, dtype=np.int32)
    proposal_bbox_targets=np.zeros((0, 4*cfg.NUM_CLASSES), dtype=np.float32)
    all_bbox_weights=np.zeros((0, 4*cfg.NUM_CLASSES), dtype=np.float32)
    all_proposals=np.zeros((0, 5),dtype=np.float32)
    
    assert isinstance(gt_boxes, list), 'gt boxes must be list'
    assert isinstance(gt_classes, list), 'gt classes must be list'
    labels=[]
    
    for i in range(batch_size):
        proposals_this_image=proposal_batch[i]
        gt_boxes_this_image=gt_boxes[i]
        gt_classes_this_image=gt_classes[i]

        bbox_overlaps=bbox_overlaps_per_image(proposals_this_image, gt_boxes_this_image, branch='frcnn')
        argmax_overlaps=np.argmax(bbox_overlaps, axis=1)      

        max_overlaps=np.max(bbox_overlaps, axis=1)
        
        fg_inds=np.where(max_overlaps >= cfg[cfg.PHASE].FG_THRESH)[0]
        if cfg.CHOOSE_PROPOSAL:
            fg_rois_this_image=min(fg_rois_per_image, fg_inds.size)
        else:
            fg_rois_this_image=fg_inds.size
        
        num_fg_rois+=fg_rois_this_image
        
        if cfg.CHOOSE_PROPOSAL and fg_inds.size>0:
            fg_inds = np.random.choice(fg_inds, size=fg_rois_this_image, replace=False)
            
        bg_inds=np.where(np.bitwise_and(bbox_overlaps < cfg[cfg.PHASE].BG_THRESH_HI, bbox_overlaps >= cfg[cfg.PHASE].BG_THRESH_LO)==1)[0]
        if cfg.CHOOSE_PROPOSAL:
            bg_rois_per_image = rois_per_image - fg_rois_this_image
            bg_rois_this_image = min(bg_rois_per_image, bg_inds.size)
        else:
            bg_rois_this_image = bg_inds.size

        num_bg_rois+=bg_rois_this_image
        
        if cfg.CHOOSE_PROPOSAL and bg_inds.size > 0:
            bg_inds = np.random.choice(bg_inds, size=bg_rois_this_image, replace=False)
        
        '''discard rois with too small overlaps (less than BG_THRESH_LO)'''
        keep_inds=np.append(fg_inds, bg_inds)
 
        proposals_keep=proposals_this_image[keep_inds]
        all_proposals=np.append(all_proposals, proposals_keep, 0)
       
        max_overlap_boxes=gt_boxes_this_image[argmax_overlaps[keep_inds]]
        gt_inds=gt_classes_this_image[argmax_overlaps[keep_inds]] 
#        print(gt_inds)
        targets=bbox_transform(proposals_keep, max_overlap_boxes)
        if cfg.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            #targets = (targets - cfg.BBOX_NORMALIZE_MEANS) / cfg.BBOX_NORMALIZE_STDS
            targets = targets/cfg.BBOX_STD_DEV
        
        proposal_cls_target=np.zeros(len(proposals_keep), dtype=np.int32) 
        bbox_deltas_classes=np.zeros((len(proposals_keep), 4*cfg.NUM_CLASSES), dtype=np.float32)

        for j in range(len(keep_inds)):
            proposal_cls_target[j]=gt_inds[j]
            bbox_deltas_classes[j,gt_inds[j]*4:gt_inds[j]*4+4]=targets[j][...]
            
        bbox_weights=np.ones((proposal_cls_target.shape[0], 4*cfg.NUM_CLASSES),dtype=np.float32)
        proposal_cls_target[fg_rois_this_image:]=0
        bbox_deltas_classes[fg_rois_this_image:,:]=0
        bbox_weights[fg_rois_this_image:,:]=0

        labels_this_image=np.ones(len(keep_inds), dtype=np.int32)
        labels_this_image[:fg_rois_this_image]=gt_inds[:fg_rois_this_image]
        labels_this_image[fg_rois_this_image:]=0
        labels.append(labels_this_image)
        
        proposal_cls_targets=np.append(proposal_cls_targets, proposal_cls_target)
        proposal_bbox_targets=np.append(proposal_bbox_targets, bbox_deltas_classes, 0)
        all_bbox_weights=np.append(all_bbox_weights, bbox_weights, 0)        
#    return all_proposals, proposal_cls_targets, proposal_bbox_targets, all_bbox_weights, num_proposals, num_fgs
    return all_proposals, proposal_cls_targets, proposal_bbox_targets, all_bbox_weights, labels