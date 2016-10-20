#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Written by yzh
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

database_dir = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/data/fddb'

def run(caffe_model_dir,prototxt_dir,exp_name,RPN_POST_NUM=50,INPUT_SIZE=200):
    if '/' not in caffe_model_dir:
        caffe_model_dir='/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/'+caffe_model_dir
    if prototxt_dir[0]!='/':
        prototxt_dir='/home/yzh/Documents/caffe_workspace/py-faster-rcnn/models/pascal_voc/'+prototxt_dir
    cfg.TEST.HAS_RPN = True
    cfg.TEST.RPN_POST_NMS_TOP_N = RPN_POST_NUM
    cfg.TEST.SCALES = (INPUT_SIZE,)
    caffemodel = caffe_model_dir
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.').format(caffemodel))
    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0
    net = caffe.Net(prototxt_dir, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)
    fold_num = 0
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
            dets, im = forward(net, im_name)
            fsave.writelines(format_rst(im_file, dets[:, :4], dets[:, 4:].flatten()))
        fsave.close()
    os.system('cd /home/yzh/Documents/caffe_workspace/py-faster-rcnn/data/fddb/evaluation;'
              './runEvaluate.pl;'
              'cd /home/yzh/Documents/caffe_workspace/py-faster-rcnn/data/fddb;'
              )
    f=open('/home/yzh/Documents/caffe_workspace/py-faster-rcnn/data/fddb/rstDiscROC.txt')
    s=caffe_model_dir+'\n'
    s+=f.readline()
    for l in f.readlines():
        if ' 1000 ' in l or ' 1001 ' in l:
            s+=l
        if ' 200 ' in l or ' 201 ' in l:
            s+=l
    f=open('%s_log.txt' % (exp_name),'a')
    f.write(s)
    f.close()

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


if __name__=='__main__':
    #only for test if the script can run
    run('zf_fc128_iter_60000.caffemodel',
        'ZF/fc128/test.prototxt',
        'fc128'
        )