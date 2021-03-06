import os

rnn_seq_len=10
num_clusters=2
# model settings
model = dict(
    type='ClusterRCRNN',
    pretrained='torchvision://resnet50',
    swapped=False,
    bidirectional=True,
    rnn_level=3,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=2,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),    
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    mask_cluster_head=dict(
        type='MaskClusterHead',
        num_convs=4,
        num_fcs=2,
        roi_feat_size=14,
        in_channels=256,        
        conv_out_channels=256,
        fc_out_channels=1024,
        num_clusters=num_clusters+1,
        num_instances=100,
        max_distance=0.9,
        num_samples=20,
        roi_cluster=False,
        loss_cluster=dict(type='ConstrainedClusteringLikelihoodLoss', loss_weight=1.0),
        loss_background=dict( type='CrossEntropyLoss', no_logit=True, loss_weight=1.0)),
    roi_cluster_head=dict(
        type='MaskClusterHead',
        num_convs=4,
        num_fcs=2,
        roi_feat_size=14,
        in_channels=256+1,        
        conv_out_channels=256,
        fc_out_channels=1024,
        num_clusters=num_clusters,
        roi_cluster=True,
        num_instances=100,
        max_distance=0.9,
        loss_cluster=dict(type='ConstrainedClusteringLikelihoodLoss', loss_weight=1.0)),
    gru_model=dict(
        type='ConvGRU',
        input_size=2048,
        hidden_sizes=[256,1024], #[32, 256],
        kernel_sizes=[5, 3],
        num_layers=2))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=64,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=128,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        mask_size=28,
        pos_weight=-1,
        roi_cluster=False,
        roi_cluster_score=2.0,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=200,
        nms_post=200,
        max_num=200,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_thr=0.75),
        max_per_img=10,
        mask_thr_binary=0.5))
# dataset settings
dataset_type = 'RatsDataset'
test_subset = 'minval'
data_root = '/media/hdd/aron/rats/human/8597db4/training_data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_inst_id=True),
    dict(type='Resize', img_scale=(640, 360), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_inst_ids', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 360),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        #ann_file=data_root + 'annotations/instances_train_minusminval_filtered.json',
        #ann_file=data_root + 'annotations/instances_minival_filtered.json',
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root + 'images',
        seq_len=rnn_seq_len,
        step=5,
        pipeline=train_pipeline),        
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'val_videos',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        #ann_file=data_root + 'annotations/instances_test.json',
        #img_prefix=data_root + 'test_videos',
        #ann_file=data_root + 'annotations/instances_val.json',
        #img_prefix=data_root + 'val_videos',

        ann_file='/media/hdd/aron/rats/annotations/ann.json',
        img_prefix='/media/hdd/aron/rats/test_images',
        seq_len=rnn_seq_len,
        step=rnn_seq_len-1,
        
        #ann_file=data_root + 'annotations/instances_train_short.json',
        #img_prefix=data_root + 'short_videos',
        #ann_file=data_root + 'annotations/instances_train.json',
        #img_prefix=data_root + 'videos',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['segm'])
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/media/hdd/aron/work_dirs/luigi_work_dir_10_lr001_e12_1_001/'
#work_dir = work_dir + os.listdir(work_dir)[0]
#load_from = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth'
load_from = 'work_dirs/luigi_work_dir_1_lr001_e12_001/epoch_12.pth'
#esume_from = work_dir + '/epoch_1.pth'
resume_from = None
workflow = [('train', 1)]
