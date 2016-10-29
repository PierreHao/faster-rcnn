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
import Image
import numpy as np

RPN_POST_NUM=50
INPUT_SIZE=600
cfg.TEST.RPN_POST_NMS_TOP_N = RPN_POST_NUM
cfg.TEST.SCALES = (INPUT_SIZE,)
cfg.TEST.RPN_NMS_THRESH = 0.7

default_stting={
    'rpn_post_num':50,
    'input_size':600,
    'rpn_nms_thresh':0.5
}

class CascadeModel():
    def __init__(self):
        self.models=[]
        self.settings=[]

    def add_facemodel(self,caffemodel,prototxt,setting=default_stting):
        self.models.append(FaceModel(caffemodel,prototxt,cpu=1))
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

    def _gen_im(self,im,dets):
        im_dirs=[]
        if len(dets)==0:return im_dirs
        dets=self._enlarge_dets(dets[:,:4],im)
        for i,det in enumerate(dets):
            crop_im=im.crop(det)
            direction='temp/temp%f.jpg'%(time.time(),)
            crop_im.save(direction)
            im_dirs.append(direction)
        return im_dirs

    def _clear_temp(self,im_name):
        if im_name[:4]=='temp':
            os.remove(im_name)

    def _clear_all_temp(self):
        for name in os.listdir('temp'):
            os.remove('temp/'+name)

    def forward(self,image_path):
        images=[image_path]
        ims=[]
        for i,model in enumerate(self.models):
            setting=self.settings[i]
            cfg.TEST.RPN_POST_NMS_TOP_N = setting['rpn_post_num']
            cfg.TEST.SCALES = (setting['input_size'],)
            cfg.TEST.RPN_NMS_THRESH=(setting['rpn_nms_thresh'])
            new_images=[]
            if i==len(self.models)-1:
                model.vis=1
            for path in images:
                im,dets=model.forward(path)
                new_images+=self._gen_im(im,dets)
                self._clear_temp(path)
                if i == len(self.models) - 1:
                    ims.append(im)
            images=new_images
        self._clear_all_temp()
        return ims






if __name__ == '__main__':
    #only to test if the FaceModel can work correctly
    caffe_model_dir='/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/fc6_512_no_pool_iter_60000.caffemodel'
    prototxt='/home/yzh/Documents/caffe_workspace/py-faster-rcnn/models/pascal_voc/ZF/fc6_512_no_pool/test.prototxt'
    model=CascadeModel()
    model.add_facemodel(caffe_model_dir,prototxt=prototxt)
    setting2={
        'rpn_post_num':50,
        'input_size':100,
        'rpn_nms_thresh':0.5
    }
    model.add_facemodel(caffe_model_dir,prototxt=prototxt,setting=setting2)
    dir_name='/home/yzh/Documents/caffe_workspace/py-faster-rcnn/data/demo/otureo/'
    for im_name in os.listdir(dir_name):
        if '.jpg' not in im_name:
            continue
        ims=model.forward (dir_name+im_name)
        for im in ims:
            im.show()

