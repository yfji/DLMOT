import numpy as np
import cv2

def resize_and_pad_image(image, im_w, im_h):
    h,w=image.shape[0:2]
    scale_x=1.0*im_w/w
    scale_y=1.0*im_h/h
    '''align the longest side'''
    scale=min(scale_x, scale_y)
    
    image_scale=cv2.resize(image, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    hs,ws=image_scale.shape[0:2]
    
    image_pad=128*np.ones((im_h,im_w,3),dtype=np.uint8)
    start_x=(im_w-ws)//2
    start_y=(im_h-hs)//2
    image_pad[start_y:start_y+hs,start_x:start_x+ws,:]=image_scale[...]

    return image_pad,(start_x, start_y),scale