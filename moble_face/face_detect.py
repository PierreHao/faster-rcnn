#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg,cfg_from_file
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import Image,ImageDraw
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

save_pic=0
show=1
cpu=0
CONF_THRESH = 0.6
print_thresh= 0.4

exp_name='ofc6_256_no_pool_SF'
save_rst=1

model='zf'
if model=='zf':
    caffe_model_dir='/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/face_zf/voc_2007_trainval/' \
                    '%s_iter_60000.caffemodel'%exp_name
    cfg_file = 'experiments/cfgs/face_zf.yml'
if model == 'googlenet':
    caffe_model_dir = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/' \
                    '%s_iter_60000.caffemodel'%exp_name
    cfg_file = 'experiments/cfgs/face_googlenet.yml'

im_names=['/home/yzh/tmp/2.jpg','/home/yzh/tmp/3.jpg']


NETS = {'vgg16': ('VGG16',),
        'zf': ('ZF',),
        'googlenet': ('Googlenet',)}
database_dir = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/data/fddb'
CLASSES = ('__background__',
           'face')
prototxt = os.path.join(cfg.MODELS_DIR, NETS[model][0],
                        exp_name, 'test.prototxt')

def vis_detections(im, dets,im_name, thresh=CONF_THRESH):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        pass
    im = im[:, :, (2, 1, 0)]
    im=Image.fromarray(im)
    draw=ImageDraw.Draw(im)
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        draw.rectangle(bbox)
        draw.text(bbox[:2],str(score))
    if save_pic:
        save_dir=os.path.join(cfg.DATA_DIR,'demo','end2end',os.path.split(im_name)[-1])
        print "save pic %s"%save_dir
        im.save(save_dir)
    if show:
        im.show()
        a=raw_input()

tot_time=0
tot_num=0
def forward(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    global tot_time,tot_num
    tot_time+=timer.total_time
    tot_num+=1
    if tot_num==2:
        tot_time=timer.total_time*2
    print ('Detection took {:.3f}s for '
           'Average {:.3f}s').format(timer.total_time, tot_time/tot_num)

    # Visualize detections for each class

    NMS_THRESH = 0.3

    cls_ind = 1
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    return dets,im


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default=model)

    args = parser.parse_args()

    return args

def format_rst(im, boxs, scores):
    im_scale=1
    boxs=boxs.tolist()
    scores=scores.tolist()
    im = im.split('image/')[-1]
    lines = [im]
    if im[-4:] == '.jpg':
        lines[0] = im[:-4]  # remove the .jpg
    lines.append(str(len(boxs)))
    for id, i in enumerate(boxs):
        lines.append(
            "%f %f %f %f %f" % (i[0] / im_scale,
                                i[1] / im_scale,
                                (i[2] - i[0]) / im_scale,
                                (i[3] - i[1]) / im_scale,
                                scores[id]))
    return [i + '\n' for i in lines]

if __name__ == '__main__':

    args = parse_args()

    caffemodel = caffe_model_dir

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.').format(caffemodel))

    if cfg_file is not None:
        cfg_from_file(os.path.join(cfg.ROOT_DIR, cfg_file))
    cfg.GPU_ID = args.gpu_id

    if cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    print prototxt
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    for im_name in im_names:
        dets, im = forward(net, im_name)
        vis_detections(im, dets, im_name)



