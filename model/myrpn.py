import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class RPN(nn.Module):
    def __init__(self, in_ch, out_ch, cls_out_ch, bbox_out_ch):
        super(RPN, self).__init__()
        self.in_ch=in_ch
        self.rpn_out_ch=out_ch
        self.cls_out_ch=cls_out_ch
        self.bbox_out_ch=bbox_out_ch
    
        self.make_rpn()
        
    def make_rpn(self):
        self.padding=SamePad2d(kernel_size=3, stride=1)
        self.shared_conv=nn.Conv2d(self.in_ch, self.rpn_out_ch, 3, 1, padding=0)
        self.rpn_cls_conv=nn.Conv2d(self.rpn_out_ch, self.cls_out_ch, 1, 1, padding=0)
        self.rpn_bbox_conv=nn.Conv2d(self.rpn_out_ch, self.bbox_out_ch, 1, 1, padding=0)
    
    def forward(self, in_tensor):
        x=F.relu(self.shared_conv(self.padding(in_tensor)))
        rpn_logits=self.rpn_cls_conv(x)
        rpn_bbox=self.rpn_bbox_conv(x)        
        return rpn_logits, rpn_bbox

class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__