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

class FaceModel():
    def __init__(self,caffemodel,prototxt,gpu_id=0,cpu=0):
        """
        init the model
        :param caffemodel: the caffemodel path.
        :param prototxt: the test prototxt path.
        """
        if not os.path.isfile(caffemodel):
            raise IOError(('{:s} not found.').format(caffemodel))
        self.caffemodel_dir=caffemodel
        if not os.path.isfile(prototxt):
            raise IOError(('{:s} not found.').format(prototxt))
        self.prototxt_dir = prototxt
        self.vis=0
        self.cpu=cpu
        self.gpu_id=gpu_id
        self.init_caffe()

    def init_caffe(self):
        #initial the caffe
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        if self.cpu:
            caffe.set_mode_cpu()
            cfg.USE_GPU_NMS=0
        else:
            caffe.set_mode_gpu()
            caffe.set_device(self.gpu_id)
            cfg.GPU_ID = self.gpu_id
        self.net = caffe.Net(self.prototxt_dir, self.caffemodel_dir, caffe.TEST)
        print '\n\nLoaded network from {:s}'.format(self.caffemodel_dir)

    def forward(self,image_path):
        """ run face detection for the image at 'image_path'
            :return the PIL Image object with the proposal and detection result
        """
        if not os.path.isfile(image_path):
            print "image is not found at: %s"%(image_path)
            return
        im_name=image_path
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Run forward for {}'.format(im_name)
        im,dets=self._detect(im_name)
        if self.vis and len(dets)>0:
            im=self._vis_detection(im,dets)
        else:
            im=Image.fromarray(im[:, :, (2, 1, 0)])
        return im,dets

    def _detect(self,im_name):
        """Detect face in an image,return Image and dets [num_bbox,5(box,score)]
        """
        im = cv2.imread(im_name)
        timer = Timer()
        timer.tic()
        # scores[R,2],boxes[R,8],including the background
        scores, boxes = im_detect(self.net, im)
        timer.toc()
        print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3

        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, 1:2]
        dets = np.hstack((cls_boxes,cls_scores)).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        cutted_dets = np.array([i for i in dets if i[-1] > CONF_THRESH])
        return im,cutted_dets

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
    caffe_model_dir='/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/fc6_512_no_pool_iter_60000.caffemodel'
    prototxt='/home/yzh/Documents/caffe_workspace/py-faster-rcnn/models/pascal_voc/ZF/fc6_512_no_pool/test.prototxt'
    model=FaceModel(caffe_model_dir,prototxt=prototxt,cpu=1)
    model.vis=1
    im, dets = model.forward ('/home/yzh/Documents/caffe_workspace/py-faster-rcnn/data/demo/1.jpg')
    im.show()
