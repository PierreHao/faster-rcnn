#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN for face_detection
# Written by yuanzhihang1@126.com
# --------------------------------------------------------
import sys, os

sys.path.insert(0,'tools')

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from rpn.generate import imdb_proposals
import argparse
import pprint
import numpy as np

import multiprocessing as mp
import cPickle
import shutil

exp_name='conv41ofc_2'
pre_train_name='conv41ofc_2'
use_last_train=1
last_iter=40000


model='ZF'
stage = []

if model=='googlenet':
    pretrained_model = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/face_googlenet/voc_2007_trainval/' \
                       '%s_iter_60000.caffemodel' % (pre_train_name)
    pretrained_model = '/home/yzh/caffe/models/googlenet/GoogleNet_SOS.caffemodel'
    cfg_file = 'experiments/cfgs/face_googlenet.yml'
elif model=='ZF':
    pretrained_model = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/data/imagenet_models/ZF.v2.caffemodel'
    if use_last_train:
        pretrained_model = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/face_zf/voc_2007_trainval/' \
                       '%s_iter_%d.caffemodel' % (pre_train_name,last_iter)
    cfg_file = 'experiments/cfgs/face_zf.yml'
elif model=='resnet':
    pretrained_model = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/data/imagenet_models/ResNet-50-model.caffemodel'
    if use_last_train:
        pretrained_model = '/home/yzh/Documents/caffe_workspace/py-faster-rcnn/output/face_resnet/voc_2007_trainval/' \
                       '%s_iter_%d.caffemodel' % (pre_train_name,last_iter)
    cfg_file = 'experiments/cfgs/face_resnet.yml'

net_name=model+'/'+exp_name
stage.append(pretrained_model)

max_iters=80000

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
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

    if cfg_file is not None:
        cfg_from_file(os.path.join(cfg.ROOT_DIR,cfg_file))
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
