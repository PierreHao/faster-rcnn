# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps

DEBUG = False

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._num_classes = layer_params['num_classes']

        #top: 'cls_score_OHEM'
        top[0].reshape(1, self._num_classes)
        #top: 'labels_OHEM'
        top[1].reshape(1, 1)
        #top: 'bbox_targets_OHEM'
        top[2].reshape(1, self._num_classes * 4)
        #top: 'bbox_pred_OHEM'
        top[3].reshape(1, self._num_classes * 4)
        #top: 'bbox_inside_weights_OHEM'
        top[4].reshape(1, self._num_classes * 4)
        #top: 'bbox_outside_weights_OHEM'
        top[5].reshape(1, self._num_classes * 4)
        #top: 'rois_OHEM'
        top[6].reshape(1,self._num_classes*4)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        bbox_pred = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data

        cls_score = bottom[2].data

        #calculate overlaps
        overlaps = bbox_overlaps(
            np.ascontiguousarray(bbox_pred[:, 1:5], dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)


        #sift the boxs
        hard_indexs=[]
        labels=[]
        for i,max_overlap in enumerate(max_overlaps):
            if max_overlap>0.5 and cls_score[i][0]>cls_score[i][1]:
                hard_indexs.append(i)
                labels.append(1)
            if max_overlap<0.5 and cls_score[i][0]<cls_score[i][1]:
                hard_indexs.append(i)
                labels.append(0)
        if len(hard_indexs)==0:
            hard_indexs=range(len(bbox_pred))
            labels=bottom[4].data
        hard_num=len(hard_indexs)
        for i in xrange(len(bottom[7].data)):
            if i >= hard_num:
                hard_indexs.append(hard_indexs[i % hard_num])
                labels.append(labels[i % hard_num])
        labels = np.array(labels, dtype=np.float32)

        bbox_target_data = _compute_targets(
            bbox_pred[hard_indexs, 1:5], gt_boxes[gt_assignment[hard_indexs], :4], labels)

        bbox_targets, bbox_inside_weights = \
            _get_bbox_regression_labels(bbox_target_data, self._num_classes)


        hard_num=len(bottom[7].data)
        # top: 'cls_score_OHEM'
        top[0].reshape(hard_num,cls_score.shape[1])
        top[0].data[...]=cls_score[hard_indexs]
        #top[0].data[...]=cls_score#[hard_indexs]

        #top: 'labels_OHEM'
        top[1].reshape(hard_num)
        top[1].data[...]=labels
        #top[1].data[...]=bottom[4].data

        #top: 'bbox_targets_OHEM'
        top[2].reshape(hard_num,self._num_classes * 4)
        top[2].data[...] = bbox_targets
        #top[2].data[...] = bottom[3].data

        #top: 'bbox_pred_OHEM'
        top[3].reshape(hard_num, self._num_classes * 4)
        top[3].data[...] = bbox_pred[hard_indexs]
        #top[3].data[...] = bbox_pred#[hard_indexs]

        #top: 'bbox_inside_weights_OHEM'
        top[4].reshape(hard_num, self._num_classes*4)
        top[4].data[...] = bbox_inside_weights
        #top[4].data[...] = bottom[5].data

        #top: 'bbox_outside_weights_OHEM'
        top[5].reshape(hard_num, self._num_classes * 4)
        top[5].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)
        #top[5].data[...] = bottom[6].data

        #top: 'rois_OHEM'
        top[6].reshape(hard_num, 5)
        top[6].data[...] = bottom[7].data[hard_indexs]


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights
