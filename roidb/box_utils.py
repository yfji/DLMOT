import numpy as np
from rpn.util import bbox_overlaps_per_image
import rpn.generate_anchors as G
import random
'''
RPN
raw image level
'''
def calc_search_boxes(temp_boxes, bound):
    search_boxes=np.zeros((0,4),dtype=temp_boxes.dtype)
    for temp_box in temp_boxes:
        x1,y1,x2,y2=temp_box.tolist()
        cx=0.5*(x1+x2)
        cy=0.5*(y1+y2)
        w=x2-x1+1
        h=y2-y1+1
        shift_w=w*1
        shift_h=h*1
        xmin=max(0,cx-shift_w)
        ymin=max(0,cy-shift_h)
        xmax=min(bound[0], cx+shift_w-1)
        ymax=min(bound[1], cy+shift_h-1)
        search_boxes=np.append(search_boxes, np.asarray([[xmin,ymin,xmax,ymax]]), 0)
        
    return search_boxes

def filter_boxes(boxes):
    x1,y1,x2,y2=np.split(boxes, 4, axis=1)

    ws=x2-x1+1
    hs=y2-y1+1

    filter_inds=np.where(np.bitwise_and(ws>16,hs>16)==1)[0]
    return filter_inds

def align_boxes(ref_boxes, det_boxes):
    ref_ids=list(ref_boxes.keys())
    det_ids=list(det_boxes.keys())

    ref_boxes_align = []
    det_boxes_align = []

    if len(ref_ids)==0:
        print('Empty ref_ids')
    else:
        ref_ids=np.sort(ref_ids)
        det_ids=np.sort(det_ids)

        for ref_id in ref_ids:
            ref_boxes_align.append(ref_boxes[ref_id])
            if ref_id not in det_ids:
                det_boxes_align.append(np.zeros(4, dtype=np.float32))
            else:
                det_boxes_align.append(det_boxes[ref_id])

    if len(ref_boxes_align)==0:
        ref_boxes_align=np.zeros((0,4))
    else:
        ref_boxes_align = np.vstack(ref_boxes_align)
    if len(det_boxes_align)==0:
        det_boxes_align = np.zeros((0, 4))
    else:
        det_boxes_align=np.vstack(det_boxes_align)
    
    if len(ref_ids)>0:
        inds=filter_boxes(ref_boxes_align)
        ref_boxes_align=ref_boxes_align[inds]
        det_boxes_align=det_boxes_align[inds]

    return ref_boxes_align, det_boxes_align

def clip_bboxes(bboxes, w, h):
    _bboxes=bboxes.copy()
    _bboxes[:,0]=np.maximum(0, _bboxes[:,0])
    _bboxes[:,1]=np.maximum(0, _bboxes[:,1])
    _bboxes[:,2]=np.minimum(w-1, _bboxes[:,2])
    _bboxes[:,3]=np.minimum(h-1, _bboxes[:,3])
    return _bboxes
    
def best_search_box_test(templates, temp_box, template_anchors, bound):    
    tmpl_sz=templates[:,0]
    tmpl_inds=np.arange(len(templates))

    x1,y1,x2,y2=temp_box[:]
    tw,th=x2-x1+1,y2-y1+1

    cx, cy=x1+0.5*tw, y1+0.5*th
#    sz=np.sqrt(tw*th)
    sz=max(tw,th)
    if 2*sz<tmpl_sz[0]:
        ind1=np.array([0])
    else:
        ind1=np.where(np.bitwise_and(tmpl_sz>=sz, tmpl_sz<=2.2*sz))[0]
       
    if ind1.size==0:
        print(templates)
        print(temp_box)
        assert ind1.size>0, 'Ind1 weird error. Det box size larger than image size'

    if ind1.size==1:
        best_ind=ind1[0]
        best_sz_half=templates[best_ind,0]*0.5
        cx=min(max(cx, best_sz_half), bound[0]-best_sz_half)
        cy=min(max(cy, best_sz_half), bound[1]-best_sz_half)
        choice_tmpl=np.array([cx-best_sz_half, cy-best_sz_half, cx+best_sz_half, cy+best_sz_half], dtype=np.float32)
        anchors=template_anchors[best_ind]
        anchors[:,[0,2]]+=choice_tmpl[0]
        anchors[:,[1,3]]+=choice_tmpl[1]
        return choice_tmpl, anchors
    
    templates=templates[ind1]
    tmpl_sz=tmpl_sz[ind1]
    tmpl_inds=tmpl_inds[ind1]
    _anchors=template_anchors[ind1]
    
    rat=np.abs(1.0*tmpl_sz/sz-1)
    
    best_ind=np.argmin(rat)
    best_tmpl=templates[best_ind]
    anchors=_anchors[best_ind]
    
    best_sz_half=best_tmpl[0]*0.5

    cx=min(max(cx, best_sz_half), bound[0]-best_sz_half)
    cy=min(max(cy, best_sz_half), bound[1]-best_sz_half)
    
    choice_tmpl=np.array([cx-best_sz_half, cy-best_sz_half, cx+best_sz_half, cy+best_sz_half], dtype=np.float32)
    anchors[:,[0,2]]+=choice_tmpl[0]
    anchors[:,[1,3]]+=choice_tmpl[1]
    
    return choice_tmpl, anchors

def best_search_box_random(temp_box, det_box, templates, template_anchors, bound, fg_thresh):
    tmpl_sz=templates[:,0]
    tmpl_inds=np.arange(len(templates))

    x1,y1,x2,y2=temp_box[:]
    tw,th=x2-x1+1,y2-y1+1

    cx, cy=x1+0.5*tw, y1+0.5*th
    sz=max(tw,th)
    #sz=np.sqrt(tw*th)

    if 2*sz<tmpl_sz[0]:
        inds=np.array([0])
    else:
        ind2=np.bitwise_and(tmpl_sz>=sz, tmpl_sz<=2*sz)
        inds=np.where(ind2==1)[0]

    if inds.size==0:
        print(templates)
        print(temp_box)
        assert inds.size>0, 'Inds weird error. Det box size larger than image size'

    tmpls=templates[inds]
    tmpl_inds=tmpl_inds[inds]
    
    choice_ind=np.random.choice(np.arange(inds.size))
    choice_tmpl=tmpls[choice_ind]
    choice_sz_half=choice_tmpl[0]*0.5

    cx=min(max(cx, choice_sz_half), bound[0]-choice_sz_half)
    cy=min(max(cy, choice_sz_half), bound[1]-choice_sz_half)

    choice_tmpl=np.array([cx-choice_sz_half, cy-choice_sz_half, cx+choice_sz_half, cy+choice_sz_half], dtype=np.float32)
    
    anchors=template_anchors[inds][choice_ind]
    anchors[:,[0,2]]+=choice_tmpl[0]
    anchors[:,[1,3]]+=choice_tmpl[1]

    overlaps=bbox_overlaps_per_image(anchors, det_box.reshape(1,-1))    #[N,1]
    fg_inds=np.where(overlaps.ravel()>=fg_thresh)[0]

    if fg_inds.size==0:
        return None

    return choice_tmpl, overlaps, anchors

'''
deprecating...
'''
def best_search_box_train(temp_box, det_box, templates, template_anchors, bound, fg_thresh):
    tmpl_sz=templates[:,0]

    x1,y1,x2,y2=temp_box[:]
    tw,th=x2-x1+1,y2-y1+1
    cx, cy=x1+0.5*tw, y1+0.5*th

    ind1=np.where(np.bitwise_and(tmpl_sz>=tw, tmpl_sz>=th)==1)[0]     
    _templates=templates[ind1]
    _tmpl_anchors=template_anchors[ind1]

    overlaps_list=[]
    shift_tmpls=[]
    anchors_list=[]
    for i, tmpl in enumerate(_templates):
        sz_half=tmpl[0]*0.5
        cx=min(max(cx, sz_half), bound[0]-sz_half)
        cy=min(max(cy, sz_half), bound[1]-sz_half)
        shift_tmpl=np.array([cx-sz_half, cy-sz_half, cx+sz_half, cy+sz_half], dtype=np.float32)
        _anchors=_tmpl_anchors[i]
        _anchors[:,[0,2]]+=shift_tmpl[0]
        _anchors[:,[1,3]]+=shift_tmpl[1]

        overlaps=bbox_overlaps_per_image(_anchors, det_box.reshape(1,-1)).ravel()
        fg_inds=np.where(overlaps>=fg_thresh)[0]
        overlaps_list.append(fg_inds.size)
        shift_tmpls.append(shift_tmpl)
        anchors_list.append(_anchors)
    
    overlaps_list=np.array(overlaps_list)
    best_ind=np.argmax(overlaps_list)
    max_overlaps_num=overlaps_list[best_ind]

    replicas=np.where(overlaps_list==max_overlaps_num)[0]
    if replicas.size==1:
        best_tmpl=shift_tmpls[best_ind]
        return best_tmpl, overlaps_list[best_ind], anchors_list[best_ind]
    else:
        if replicas.size==0:
            print(overlaps_list)
            print(max_overlaps_num)
            print(replicas)
            assert 0,'Weird error, replicas is zero'
        best_tmpl, best_anchors=best_search_box_test(_templates, temp_box, _tmpl_anchors, bound)
        return best_tmpl, max_overlaps_num, best_anchors