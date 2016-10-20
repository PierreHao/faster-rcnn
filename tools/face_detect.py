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
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import numpy as np
from utils.timer import Timer
import Image,ImageDraw
import caffe, os, sys, cv2
import argparse

RPN_POST_NUM=50
INPUT_SIZE=600
cfg.TEST.RPN_POST_NMS_TOP_N = RPN_POST_NUM
cfg.TEST.SCALES = (INPUT_SIZE,)

CLASSES = ('__background__',
           'face')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
               'ZF_faster_rcnn_final.caffemodel'),
        'face': ('VGG16',
                 "VGG16_faster_rcnn_face.caffemodel")}

class FaceModel():
    def __init__(self,caffemodel,gpu_id=0,cpu=0,prototxt=''):
        """
        init the model
        :param caffemodel: the caffemodel path.
        """
        if not os.path.isfile(caffemodel):
            raise IOError(('{:s} not found.\nDid you run ./data/script/'
                    'fetch_faster_rcnn_models.sh?').format(caffemodel))
        if 'zf' in caffemodel:
            self.model='zf'
        else:
            self.model='vgg16'
        self.caffe_model_dir=caffemodel
        self.save_pic=1
        self.show=0
        self.cpu=cpu
        self.gpu_id=gpu_id
        #initial the caffe
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        if prototxt=='':
            self.prototxt = os.path.join(cfg.MODELS_DIR, NETS[self.model][0],
                                'faster_rcnn_end2end', 'test.prototxt')
        else:
            self.prototxt=prototxt
        if self.cpu:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(self.gpu_id)
            cfg.GPU_ID = self.gpu_id
        self.net = caffe.Net(self.prototxt, self.caffe_model_dir, caffe.TEST)
        print '\n\nLoaded network {:s}'.format(caffemodel)

    def forward(self,image_path):
        """ run face detection for the image at 'image_path'
            :return the PIL Image object with the proposal and detection result
        """
        if not os.path.isfile(image_path):
            print "image is not found at: %s"%(image_path)
            return
        im_name=image_path
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        im,dets=self._demo(self.net, im_name)
        im=self._vis_detection(im,dets)
        return im,dets

    def _demo(self,net,im_name):
        """Detect object classes in an image using pre-computed object proposals.
        """
        im = cv2.imread(im_name)
        timer = Timer()
        timer.tic()
        # Detect all object classes and regress object bounds
        scores, boxes = im_detect(self.net, im)
        timer.toc()
        print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
        return im,dets

    def _vis_detection(self,im,dets,thresh=0.1):
        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return im
        im = im[:, :, (2, 1, 0)]
        im=Image.fromarray(im)
        draw=ImageDraw.Draw(im)
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            draw.rectangle(bbox)
            draw.text(bbox[:2],str(score))
        return im

    def im_save(self,im,name):
        """ Save the image to the demo/out direction with the 'name'
        """
        if not os.path.exists(os.path.join(cfg.DATA_DIR,'demo','out')):
            os.mkdir(os.path.join(cfg.DATA_DIR,'demo','out'))
        im.save(os.path.join(cfg.DATA_DIR,'demo','out',name))


if __name__ == '__main__':
    #only to test if the FaceModel can work correctly
    caffe_model_dir='/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/face_zf/voc_2007_trainval/ofc6_256_no_pool_SF_iter_60000.caffemodel'
    prototxt='/home/yzh/Documents/caffe_workspace/py-faster-rcnn/models/pascal_voc/ZF/ofc6_256_no_pool_SF/test.prototxt'
    model=FaceModel(caffe_model_dir,prototxt=prototxt,cpu=1)
    im, dets = model.forward('/home/yzh/Documents/caffe_workspace/py-faster-rcnn/data/demo/001150.jpg')
    score_thresh=0.75
    cutted_dets=np.array([i for i in dets if i[-1]>score_thresh],dtype=int)
    #im.show()

