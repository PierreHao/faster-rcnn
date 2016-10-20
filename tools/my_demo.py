#!/usr/bin/env python
# --------------------------------------------------------
# Written by yuanzhihang1@126.com
# --------------------------------------------------------
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect, im_detect_face
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import Image, ImageDraw
from utils.cython_bbox import bbox_overlaps

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
               'ZF_faster_rcnn_final.caffemodel'),
        'face': ('VGG16',
                 "VGG16_faster_rcnn_face.caffemodel")}
CLASSES = ('__background__',
           'person')
database_dir = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets'
demo_images = os.listdir(database_dir)
save_num = 500
is_show = 0
cpu_mode = 0

model = 'VGG16'
if model == 'VGG16':
    rpn_test = 1
    caffe_model_dir = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/faster_rcnn_alt_opt/voc_2007_trainval/' \
                      'vgg16_rpn_stage1_iter_10000.caffemodel'
    prototxt_dir = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/models/pascal_voc/VGG16/face/rpn_test.pt'
elif model == 'ZF':
    rpn_test = 1
    caffe_model_dir = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/faster_rcnn_alt_opt/voc_2007_trainval/' \
                      'zf_rpn_stage1_iter_40000.caffemodel'
    prototxt_dir = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/models/pascal_voc/ZF/face/rpn_test.pt'
elif model == 'ZF_final':
    rpn_test = 0
    caffe_model_dir = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/faster_rcnn_alt_opt/voc_2007_trainval/' \
                      'zf_rpn_stage1_iter_40000.caffemodel'  # 'ZF_faster_rcnn_final.caffemodel'

    prototxt_dir = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/models/pascal_voc/ZF/face/faster_rcnn_test.pt'

threshold = .95


def boxs_merge(boxes, overlapThresh=0.2):
    if len(boxes) == 0:
        return boxes
    boxes = np.array(boxes)
    result_boxes = []
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(boxes[:, 4])

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # area of i.
        area_i = np.maximum(0, x2[i] - x1[i] + 1) * np.maximum(0, y2[i] - y1[i] + 1)
        area_array = np.zeros(len(idxs) - 1)
        area_array.fill(area_i)
        # compute the ratio of overlap
        # overlap = (w * h) / (area[idxs[:last]]  - w * h + area_array)

        overlap = (w * h) / (area[idxs[:last]])
        delete_idxs = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        xmin = 10000
        ymin = 10000
        xmax = 0
        ymax = 0
        ave_prob = 0
        width = x2[i] - x1[i] + 1
        height = y2[i] - y1[i] + 1
        for idx in delete_idxs:
            ave_prob += boxes[idxs[idx]][4]
            if (boxes[idxs[idx]][0] < xmin):
                xmin = boxes[idxs[idx]][0]
            if (boxes[idxs[idx]][1] < ymin):
                ymin = boxes[idxs[idx]][1]
            if (boxes[idxs[idx]][2] > xmax):
                xmax = boxes[idxs[idx]][2]
            if (boxes[idxs[idx]][3] > ymax):
                ymax = boxes[idxs[idx]][3]
        if (x1[i] - xmin > 0.1 * width):
            xmin = x1[i] - 0.1 * width
        if (y1[i] - ymin > 0.1 * height):
            ymin = y1[i] - 0.1 * height
        if (xmax - x2[i] > 0.1 * width):
            xmax = x2[i] + 0.1 * width
        if (ymax - y2[i] > 0.1 * height):
            ymax = y2[i] + 0.1 * height
        result_boxes.append([xmin, ymin, xmax, ymax, ave_prob / len(delete_idxs)])
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, delete_idxs)

    # return only the bounding boxes that were picked using the
    # integer data type
    # result = np.delete(boxes[pick],np.where(boxes[pick][:, 4] < 0.9)[0],  axis=0)
    # print boxes[pick]
    return result_boxes


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--net', dest='demo_net', help='Network to use [face]',
                        choices=NETS.keys(), default='face')
    args = parser.parse_args()

    return args


def show_roi(im_file_name, scale, roidb):
    im = Image.open(im_file_name)
    im = im.resize((im.size[0] * scale, im.size[1] * scale))
    draw = ImageDraw.Draw(im)

    for roi in roidb:
        draw.rectangle(roi[1:])
    if is_show:
        im.show()
        a = raw_input()
    else:
        save_dir = os.path.join(cfg.ROOT_DIR, 'data', 'demo', 'out', os.path.split(im_file_name)[-1])
        print "saved at %s" % save_dir
        im.save(save_dir)


def format_rst(im, boxs):
    lines = [im]
    if im[-4:] == '.jpg':
        lines[0] = im[:-4]  # remove the .jpg
    lines.append(len(boxs))
    for i in boxs:
        lines.append("%d %d %d %d 1" % (i[0], i[1], i[2] - i[0], i[3] - i[1]))
    return lines


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = prototxt_dir
    caffemodel = caffe_model_dir

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
    if cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    for im_i, im_name in enumerate(demo_images):
        print 'read image %s' % im_name
        im_file = os.path.join(database_dir, im_name)
        im = cv2.imread(im_file)
        output, im_scales = im_detect_face(net, im)
        if rpn_test:
            scores = output['scores']
            rois = output['rois']
        else:
            scores = output['cls_prob']
            rois = output['bbox_pred']
        roidb = []
        for score, roi in zip(scores, rois):
            # using the threshold to select the roid
            if score[0] > threshold:
                roidb.append(roi)
                # print roi
        roidb = boxs_merge(roidb)
        show_roi(im_file, im_scales, roidb)
        # a=raw_input()
        if im_i > save_num:
            break
