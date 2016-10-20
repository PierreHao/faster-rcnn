#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN for face_detection
# Written by yuanzhihang1@126.com
# --------------------------------------------------------

"""Train a Faster R-CNN network using alternating optimization.
This tool implements the alternating optimization algorithm described in our
NIPS 2015 paper ("Faster R-CNN: Towards Real-time Object Detection with Region
Proposal Networks." Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.)
"""
import _init_paths
from fast_rcnn.train import get_training_roidb, train_net,train_net_face
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from rpn.generate import imdb_proposals
import argparse
import pprint
import numpy as np
import sys, os
import multiprocessing as mp
import cPickle
import shutil

exp_name='fc6_512_conv_cut_roi_cut'

model='ZF'
if model=='VGG16':
    stage=[]
    #pretrained_model='/home/yzh/Documents/caffe_workspace/py-faster-rcnn/data/imagenet_models/VGG16.v2.caffemodel'
    pretrained_model='/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/' \
                     'vgg16_faster_0723_2.caffemodel'
    net_name='VGG16/face_end2end'
    stage.append(pretrained_model)
    pretrained_model2 = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/' \
                       'vgg16_faster_rcnn_iter_60000.caffemodel'
    stage.append(pretrained_model2)
    default_net='VGG16'
else:
    stage = []
    pretrained_model = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/data/imagenet_models/ZF.v2.caffemodel'
    #pretrained_model = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/' \
    #                   'zf_faster_rcnn_iter_20000.caffemodel'
    default_net = 'ZF'
    net_name='ZF/'+exp_name
    stage.append(pretrained_model)

max_iters=60000

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--net_name', dest='net_name',
                        help='network name (e.g., "ZF")',
                        default=default_net, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default="experiments/cfgs/faster_rcnn_end2end.yml", type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)


    args = parser.parse_args()
    return args

def _init_caffe(cfg):
    """Initialize pycaffe in a training process.
    """

    import caffe
    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

def get_roidb(imdb_name, rpn_file=None):
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    if rpn_file is not None:
        imdb.config['rpn_file'] = rpn_file
    roidb = get_training_roidb(imdb)
    return roidb, imdb

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(os.path.join(cfg.ROOT_DIR,args.cfg_file))
    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    roidb, imdb = get_roidb(args.imdb_name)
    print '{:d} roidb entries'.format(len(roidb))
    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)



    _init_caffe(cfg)

    for i,pretrained_model in enumerate(stage):
        if i!=0:
            solver = os.path.join(cfg.MODELS_DIR, net_name, 'solver%d.prototxt'%(i+1,))
        else:
            solver = os.path.join(cfg.MODELS_DIR, net_name, 'solver.prototxt')
        train_net(solver, roidb, output_dir,
                  pretrained_model=pretrained_model,
                  max_iters=max_iters)