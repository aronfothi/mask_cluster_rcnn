import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS

from .bbox_head import BBoxHead


@HEADS.register_module
class ReidBBoxHead(BBoxHead):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 with_reid = True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=81,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 reid_dim = 0,
                 reid_per_class = False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),                 
                 loss_embedding=dict(type='BatchHard',
                                use_sigmoid=False,
                                loss_weight=1.0)
                 ):
        super(ReidBBoxHead, self).__init__(with_avg_pool=with_avg_pool,
                                            with_cls=with_cls,
                                            with_reg=with_reg,
                                            roi_feat_size=roi_feat_size,
                                            in_channels=in_channels,
                                            num_classes=num_classes,
                                            target_means=target_means,
                                            target_stds=target_stds,
                                            reg_class_agnostic=reg_class_agnostic,
                                            loss_cls=loss_cls,
                                            loss_bbox=loss_bbox )
        self.loss_embedding = build_loss(loss_embedding)
        self.with_reid = with_reid
        self.reid_dim = reid_dim
        self.reid_per_class = reid_per_class

        if self.with_reid:
            if reid_per_class:
                self.fc_reid = nn.Linear(in_channels, (num_classes - 1) * reid_dim)
            else:
                self.fc_reid = nn.Linear(in_channels, reid_dim)
        

    def init_weights(self):
        super(ReidBBoxHead, self).init_weights()
        if self.with_reid:
            nn.init.normal_(self.fc_reid.weight, 0, 0.01)
            nn.init.constant_(self.fc_reid.bias, 0)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        reid_pred = self.fc_reid(x) if self.with_reid else None
        if self.reid_per_class:
            reid_pred = reid_pred.view(-1, (self.num_classes - 1), self.reid_dim)
            best_class = tf.cast(tf.argmax(classification, axis=-1) - 1, dtype=tf.int32)
            feature_indices = tf.stack([tf.range(tf.shape(best_class)[0], dtype=tf.int32), best_class], axis=1)
            reid_pred = tf.gather_nd(reid_pred, feature_indices)
        
        return cls_score, bbox_pred, reid_pred

    def _reid_target_single(self, pos_proposals, pos_assigned_gt_inds, inst_ids, cfg):
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        inst_ids = inst_ids.cpu().numpy().astype(np.int32)
        inst_targets = inst_ids[pos_assigned_gt_inds]
        
        return inst_targets

    def get_reid_target(self, sampling_results, inst_ids_list, rcnn_train_cfg):
        
        pos_proposals_list = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds_list = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        cfg_list = [rcnn_train_cfg for _ in range(len(pos_proposals_list))]
        reid_targets = map(self._reid_target_single, pos_proposals_list,
                        pos_assigned_gt_inds_list, inst_ids_list, cfg_list)
        return reid_targets

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'embedding'))
    def loss(self,
             pid_targets,
             embedding,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,             
             reduction_override=None):
        losses = super(ReidBBoxHead, self).loss(cls_score,
                                                bbox_pred,
                                                labels,
                                                label_weights,
                                                bbox_targets,
                                                bbox_weights,
                                                reduction_override)
        if embedding is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_embedding = embedding.view(embedding.size(0), 128)[pos_inds]
            else:
                pos_embedding = embedding.view(embedding.size(0), -1, 128)[pos_inds, labels[pos_inds]]
            losses['loss_embedding'] = self.loss_embedding(
                pos_embedding,
                pid_targets[pos_inds],
                #pid_weights[pos_inds],
                avg_factor=pid_targets.size(0),
                reduction_override=reduction_override)
        return losses

    