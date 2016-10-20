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
from fast_rcnn.config import cfg
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
cpu=1
CONF_THRESH = 0.8
print_thresh= 0.4
save_max_num=300

RPN_POST_NUM=5
INPUT_SIZE=200

model='zf'
if model=='zf':
    caffe_model_dir='/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/' \
                    'zf_faster_0723.caffemodel'
if model == 'vgg16':
    caffe_model_dir = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/' \
                      'vgg16_faster_rcnn_iter_60000.caffemodel'

database_dir = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/data/fddb'
demo_images=['000001.jpg'] #demo images in data/demo

CLASSES = ('__background__',
           'face')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
               'ZF_faster_rcnn_final.caffemodel'),
        'face': ('VGG16',
                 "VGG16_faster_rcnn_face.caffemodel")}

prototxt = os.path.join(cfg.MODELS_DIR, NETS[model][0],
                        'face_end2end', 'test.prototxt')

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
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
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
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    #the base less proposal cutted
    cfg.TEST.RPN_POST_NMS_TOP_N=RPN_POST_NUM
    cfg.TEST.SCALES=(INPUT_SIZE,)

    args = parse_args()


    caffemodel = caffe_model_dir

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    fold_num=0
    for fold in os.listdir(os.path.join(database_dir, 'FDDB-folds')):
        if 'ellip' in fold:
            continue
        fold_num += 1
        print '\n---\nstart evaluate the %s\n---\n' % fold
        f = open(os.path.join(database_dir, 'FDDB-folds', fold))
        demo_images = [os.path.join(database_dir, 'image', line[:-1] + '.jpg') for line in f.readlines()]
        image_num = len(demo_images)
        fsave = open(os.path.join(database_dir, 'rst', "fold-%s-out.txt" % fold[-6:-4]), 'w')
        for im_i, im_name in enumerate(demo_images):
            print 'evaluate the %s %.2f tot:10 now:%d' % (fold, float(im_i) / image_num, fold_num)
            im_file = os.path.join(database_dir, im_name)
            if not os.path.exists(im_file):
                continue
            dets,im=forward(net, im_name)
            vis_detections(im, dets,im_name)
            fsave.writelines(format_rst(im_file,dets[:,:4],dets[:,4:].flatten()))

