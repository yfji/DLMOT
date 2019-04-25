import numpy as np
from core.config import cfg
import rpn.generate_anchors as G
from rpn.template import get_template

class DataReader(object):
    def __init__(self, im_width, im_height, batch_size=8):
        self.im_w = im_width
        self.im_h = im_height
        self.batch_size=batch_size        

        self.fetch_config()
        self.bound=(im_width, im_height)
        self.out_size=(im_width//self.stride, im_height//self.stride)

        self.K=len(self.ratios)*len(self.scales)
        self.TK=len(self.track_ratios)*len(self.track_scales)

        self.templates=get_template(min_size=cfg.TEMP_MIN_SIZE, max_size=cfg.TEMP_MAX_SIZE, num_templates=cfg.TEMP_NUM)
        print('Using multi-level templates: ')
        print(self.templates)
        
        self.raw_anchors=G.generate_anchors(self.basic_size, self.ratios, self.scales)
        dummy_search_box=np.array([[0,0,self.im_w-1,self.im_h-1]])
        self.det_anchors=G.gen_region_anchors(self.raw_anchors, dummy_search_box, self.bound, K=self.K, size=self.out_size)[0]
        self.track_raw_anchors=G.generate_anchors(self.track_basic_size, self.track_ratios, self.track_scales)

        self.template_anchors=G.gen_template_anchors(self.track_raw_anchors, self.templates, K=self.TK, size=(self.rpn_conv_size, self.rpn_conv_size))

        self.max_interval=1

    def fetch_config(self):
        self.stride=cfg.STRIDE
        self.basic_size=cfg.BASIC_SIZE
        self.ratios=cfg.RATIOS
        self.scales=cfg.SCALES
        
        self.track_basic_size=cfg.TRACK_BASIC_SIZE
        self.track_ratios=cfg.TRACK_RATIOS
        self.track_scales=cfg.TRACK_SCALES
        self.rpn_conv_size=cfg.RPN_CONV_SIZE

    def enum_sequences(self):
        self.images_per_seq=np.zeros(0, dtype=np.int32)
        self.index_per_seq=np.zeros(0, dtype=np.int32)
        self.start_per_seq=np.zeros(0, dtype=np.int32)
        self.num_images=0

        self.upper_bound_per_seq=self.images_per_seq-self.batch_size-1

    def __len__(self):
         return (self.num_images//self.batch_size*self.batch_size)

    def __getitem__(self, index):
        c=index-self.start_per_seq
        c=c[np.where(c>=0)[0]]
        seq_index=np.argmin(c)

        ref_ind=index-self.start_per_seq[seq_index]
        if ref_ind==self.images_per_seq[seq_index]-1:
             ref_ind-=self.max_interval
             
        interval=np.random.randint(1,self.max_interval+1)
        det_ind=ref_ind+interval
        det_ind=min(det_ind, self.images_per_seq[seq_index]-1)

        roidb=self._get_roidb(seq_index, ref_ind, det_ind)

        return roidb

    def _get_roidb(self, deq_index, ref_ind, det_ind):
        raise NotImplementedError

    def _parse_all_anno(self):
        raise NotImplementedError