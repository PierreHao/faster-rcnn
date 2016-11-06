#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Written by yuanzhihang1@126.com
# --------------------------------------------------------

"""
The FaceModel class for face detection using Faster-rcnn.
Using Example:
imoprt face_detect
model=face_detect.FaceModel(caffe_model_dir)
im,dets=model.forward(image_dir)
im.show()
"""

import _init_paths
from fast_rcnn.config import cfg
from  face_detect import FaceModel
import os,time
import Image,ImageDraw
import numpy as np

RPN_POST_NUM=50
INPUT_SIZE=600
cfg.TEST.RPN_POST_NMS_TOP_N = RPN_POST_NUM
cfg.TEST.SCALES = (INPUT_SIZE,)
cfg.TEST.RPN_NMS_THRESH = 0.7

default_stting={
    'rpn_post_num':50,
    'input_size':400,
    'rpn_nms_thresh':0.5
}


def draw( im_path, dets, thresh=0.1):
    """Draw detected bounding boxes."""
    im = Image.open(im_path)
    if len(dets) == 0:
        return im
    draw = ImageDraw.Draw(im)
    for i in dets:
        if i[-1]<thresh:continue
        bbox = i[:4]
        score = i[-1]
        draw.rectangle(list(bbox.astype(np.int32)))
        draw.text(list(bbox[:2].astype(int)), str(score))
    return im

class CascadeModel():
    def __init__(self):
        self.models=[]
        self.settings=[]

    def add_facemodel(self,caffemodel,prototxt,setting=default_stting):
        self.models.append(FaceModel(caffemodel,prototxt))
        self.settings.append(setting)

    def _enlarge_dets(self,dets,im,factor=1.8):
        w=dets[:,2]-dets[:,0]
        h=dets[:,3]-dets[:,1]
        size=np.stack((w,h),1)
        dets1=(dets[:,:2]+dets[:,2:]-size*factor)/2
        dets2=(dets[:,:2]+dets[:,2:]+size*factor)/2
        dets1=np.maximum(0,dets1)
        dets2=np.minimum(im.size,dets2)
        return np.concatenate((dets1,dets2),1).astype(np.int32)

    def _gen_im(self,im,dets,shift):
        """generate the im and shift of the sub_im, then save them to temp dir
        im:Image
        dets:raw_dets
        shift:a list [shiftx,shifty]
        """
        im_dirs=[]
        if len(dets)==0:return im_dirs,[]
        dets=self._enlarge_dets(dets[:,:4],im)
        shifts = dets[:, :2] + shift
        for i,det in enumerate(dets):
            crop_im=im.crop(det)
            direction='temp/temp%f.jpg'%(time.time(),)
            crop_im.save(direction)
            im_dirs.append(direction)
        return im_dirs,shifts

    def _clear_temp(self,im_name):
        if im_name[:4]=='temp':
            os.remove(im_name)

    def _clear_all_temp(self):
        for name in os.listdir('temp'):
            os.remove('temp/'+name)

    def forward(self,image_path):
        images=[image_path]
        shifts=[(0,0)]
        final_dets=[]
        for i,model in enumerate(self.models):
            setting=self.settings[i]
            cfg.TEST.RPN_POST_NMS_TOP_N = setting['rpn_post_num']
            cfg.TEST.SCALES = (setting['input_size'],)
            cfg.TEST.RPN_NMS_THRESH=(setting['rpn_nms_thresh'])
            new_images=[]
            new_shifts=[]
            for path,shift in zip(images,shifts):
                im,dets=model.forward(path)
                self._clear_temp(path)
                if i == len(self.models) - 1:
                    if len(dets)!=0:
                        final_dets.append(dets+np.hstack([shift,shift,[0]]))
                else:
                    gen_ims, gen_shifts = self._gen_im(im, dets, shift)
                    new_images += gen_ims
                    new_shifts += list(gen_shifts)
                im.close()
            images=new_images
            shifts=new_shifts
        if final_dets!=[]:
            final_dets=np.concatenate(final_dets)
        self._clear_all_temp()
        return final_dets


if __name__ == '__main__':
    #only to test if the FaceModel can work correctly
    caffe_model_dir='/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/fc6_512_no_pool_iter_60000.caffemodel'
    prototxt='/home/yzh/Documents/caffe_workspace/py-faster-rcnn/models/pascal_voc/ZF/fc6_512_no_pool/test.prototxt'
    model=CascadeModel()
    model.add_facemodel(caffe_model_dir,prototxt=prototxt)
    setting2={
        'rpn_post_num':10,
        'input_size':100,
        'rpn_nms_thresh':0.5
    }
    model.add_facemodel(caffe_model_dir,prototxt=prototxt,setting=setting2)
    dir_name='/home/yzh/Documents/caffe_workspace/py-faster-rcnn/data/demo/otureo/'
    for im_name in os.listdir(dir_name):
        if '.jpg' not in im_name:
            continue
        dets=model.forward (dir_name+im_name)
        if dets==[]:
            continue
        im=draw(dir_name+im_name,dets)
        im.save(dir_name+'../otureo_out/'+im_name)

