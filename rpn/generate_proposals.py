import numpy as np
from rpn.util import bbox_transform_inv

penalty_k = 0.055
window_influence = 0.4

def top_proposals(temp_box, proposals, dist_thresh=20, topN=10, ordered=False):
    if not ordered:
        scores=proposals[:,-1]
        order=np.argsort(scores)[::-1]
        ordered_proposals=proposals[order]
    else:
        ordered_proposals=proposals
    w_box=temp_box[2]-temp_box[0]+1
    h_box=temp_box[3]-temp_box[1]+1
    
    ws=ordered_proposals[:,2]-ordered_proposals[:,0]+1
    hs=ordered_proposals[:,3]-ordered_proposals[:,1]+1
    
    temp_ctrx=temp_box[0]+w_box*0.5
    temp_ctry=temp_box[1]+h_box*0.5
    ctrx=ordered_proposals[:,0]+ws*0.5
    ctry=ordered_proposals[:,1]+hs*0.5
    
    dist_x=ctrx-temp_ctrx
    dist_y=ctry-temp_ctry
    
    keep=np.bitwise_and(np.abs(dist_x)<dist_thresh, np.abs(dist_y)<dist_thresh)
    proposals_keep=ordered_proposals[keep]

    locations=proposals_keep[:min(topN, proposals_keep.shape[0])]
    return locations
    

def best_proposal(temp_box, proposals, dist_thresh=20):
    '''already sorted??'''
    w_box=temp_box[2]-temp_box[0]+1
    h_box=temp_box[3]-temp_box[1]+1
    
    ws=proposals[:,2]-proposals[:,0]+1
    hs=proposals[:,3]-proposals[:,1]+1
    
    temp_ctrx=temp_box[0]+w_box*0.5
    temp_ctry=temp_box[1]+h_box*0.5
    ctrx=proposals[:,0]+ws*0.5
    ctry=proposals[:,1]+hs*0.5
    
    dist_x=ctrx-temp_ctrx
    dist_y=ctry-temp_ctry
    
    keep=np.where(np.bitwise_and(np.abs(dist_x)<dist_thresh, np.abs(dist_y)<dist_thresh)==1)[0]
    if keep.size==0:
        return None

    proposals_keep=proposals[keep]

    scores_keep=proposals_keep[:,-1]
    
    best_ind=np.argmax(scores_keep)
    location=proposals_keep[best_ind][np.newaxis,:]

    return location

'''
https://github.com/songdejia/Siamese-RPN-pytorch/blob/master/code/run_SiamRPN.py
'''
def best_proposal_hann(proposals, rpn_conv_size, K):
    window=np.outer(np.hanning(rpn_conv_size), np.hanning(rpn_conv_size))
    window=np.tile(window.flatten(), K)

    scores=proposals[:,-1]
    pscore = scores * (1 - window_influence) + scores* window * window_influence
    best_ind = np.argmax(pscore)

    location = proposals[best_ind][np.newaxis,:]
    return location