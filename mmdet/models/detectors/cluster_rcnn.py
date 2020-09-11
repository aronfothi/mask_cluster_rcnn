import torch
import torch.nn as nn
import numpy as np

from mmdet.core import bbox2roi, build_assigner, build_sampler, tensor2imgs, get_classes
from .two_stage import TwoStageDetector
from .. import builder
from ..registry import DETECTORS

import pycocotools.mask as mask_util
import mmcv

from mmcv import imresize

@DETECTORS.register_module
class ClusterRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,                 
                 train_cfg,
                 test_cfg,
                 backbone_of=None,
                 neck=None,
                 shared_head=None,
                 mask_head=None,
                 mask_cluster_head=None,
                 roi_cluster_head=None,
                 pretrained=None,
                 swapped=False,
                 num_clusters=2):
        super(ClusterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.swapped = swapped
        self.num_clusters = num_clusters
        if mask_cluster_head is not None:
            self.mask_cluster_head = builder.build_head(mask_cluster_head)
            self.mask_cluster_head.init_weights()

            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(mask_roi_extractor)
                self.share_roi_extractor = False

            if roi_cluster_head is not None:
                self.roi_cluster_head = builder.build_head(roi_cluster_head)
                self.roi_cluster_head.init_weights()

    @property
    def with_cluster(self):
        return hasattr(self, 'mask_cluster_head') and self.mask_cluster_head is not None

    @property
    def with_roi_cluster(self):
        return hasattr(self, 'roi_cluster_head') and self.roi_cluster_head is not None

    def forward_train(self,
                    img,
                    img_meta,
                    gt_bboxes,
                    gt_labels,
                    gt_inst_ids=None,  
                    gt_masks=None,                     
                    gt_bboxes_ignore=None,
                    proposals=None,
                    return_targets=False):
            M = self.forward_to_neck(img)
            losses, _ = self.forward_heads(  M,
                                        img_meta,
                                        gt_bboxes,
                                        gt_labels,
                                        gt_inst_ids,  
                                        gt_masks,                     
                                        gt_bboxes_ignore,
                                        proposals,
                                        return_targets)
            return losses

    def forward_to_neck(self, img, of=None):
        return self.extract_feat(img, of)

    def forward_heads(self,
                M,
                img_meta,
                gt_bboxes,
                gt_labels,
                gt_inst_ids=None,  
                gt_masks=None,                     
                gt_bboxes_ignore=None,
                proposals=None,
                return_targets=False,
                calculate_losses=True,
                img=None):

        x = M

        losses = dict()
        targets = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                            self.train_cfg.rpn)
            if calculate_losses:
                rpn_losses = self.rpn_head.loss(
                    *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
                losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            asd #TODO
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = len(img_meta)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                        gt_bboxes[i],
                                                        gt_bboxes_ignore[i],
                                                        gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                        gt_bboxes, gt_labels,
                                                        self.train_cfg.rcnn)
            if calculate_losses:
                loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                                *bbox_targets)
                losses.update(loss_bbox)

        # mask head forward and loss
        
            
        if not self.share_roi_extractor:
            pos_rois = bbox2roi(
                [res.pos_bboxes for res in sampling_results])
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], pos_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            mask_feats = bbox_feats[pos_inds]            

        pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])

        if self.with_cluster: 
            # mask cluster head forward and loss                

            mask_cluster_pred = self.mask_cluster_head(mask_feats)
            
            if gt_inst_ids==None:
                gt_inst_ids = [torch.arange(1, gt_mask.shape[0]+1) for gt_mask in gt_masks]                  
            
            mask_bg_targets, mask_inst_targets, mask_inst_score, \
            roi_inst_targets, roi_inst_score = self.mask_cluster_head.get_target( 
                                            sampling_results, gt_masks, gt_inst_ids, self.train_cfg.rcnn)              
            
            im_size_list = [max(gt_mask.shape) for gt_mask in gt_masks] 
            pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]

            if calculate_losses:
                loss_bg = self.mask_cluster_head.background_loss(mask_cluster_pred[:, :1], mask_bg_targets, pos_labels)
                losses.update(loss_bg)

            #############
            def batch_separator(assigned_gt_inds):
                start_idx = 0
                batch_sep = []
                for inds in assigned_gt_inds:
                    batch_sep.append(list(range(start_idx, start_idx+len(inds))))
                    start_idx += len(inds)
                return batch_sep

            batch_sep = batch_separator(pos_assigned_gt_inds)

            mask_cluster_pred = [mask_cluster_pred[sep] for sep in batch_sep]  
            mask_feats = [mask_feats[sep] for sep in batch_sep]          

            roi_inst_targets, roi_indeces, cluster_indeces, ext_assigned_gt_inds \
                = self.mask_cluster_head.get_roi_mask_target( 
                                            sampling_results,
                                            gt_masks,
                                            gt_inst_ids,
                                            mask_cluster_pred,
                                            self.train_cfg.rcnn)
            
            selected_cluster_featuremap = []
            ext_mask_feats = []
            for clustermap, roi_idx, cluster_idx, mask_feat in zip(mask_cluster_pred, roi_indeces, cluster_indeces, mask_feats):
                #inds = torch.arange(0, clustermap.shape[0], dtype=torch.long, device=clustermap.device)
                selected_cluster_featuremap.append(clustermap[roi_idx, cluster_idx])
                ext_mask_feats.append(mask_feat[roi_idx])
            selected_cluster_featuremap = torch.cat(selected_cluster_featuremap)
            mask_feats = torch.cat(ext_mask_feats)

            roi_cluster_pred = self.roi_cluster_head(mask_feats, selected_cluster_featuremap)

            
            ####################x
            batch_sep = batch_separator(ext_assigned_gt_inds)
            roi_cluster_pred = [roi_cluster_pred[sep] for sep in batch_sep]
            
            #roi_bboxes = [roi_bboxes[sep] for sep in batch_sep]

            if return_targets:
                roi_bboxes = [res.pos_bboxes for res in sampling_results]
                targets = dict(pos_assigned_gt_inds=pos_assigned_gt_inds,
                                ext_assigned_gt_inds=ext_assigned_gt_inds,
                                gt_bboxes=gt_bboxes,
                                im_size_list=im_size_list,
                                mask_cluster_pred=mask_cluster_pred,
                                roi_cluster_pred=roi_cluster_pred,
                                mask_inst_targets=mask_inst_targets,
                                mask_inst_score=mask_inst_score,
                                roi_bboxes=roi_bboxes,
                                mask_feats=mask_feats,
                                roi_inst_targets=roi_inst_targets,
                                roi_inst_score=roi_inst_score)
                                
        return losses, targets

    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         det_obj_ids=None,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            
            if self.swapped:
                _, mask_cluster_pred = self.mask_cluster_head(mask_feats)
                cluster_pred_pooled = torch.nn.MaxPool2d(2, 2)(mask_cluster_pred)
                mask_feats = torch.cat((mask_feats, cluster_pred_pooled), 1)
            mask_pred = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(mask_pred, _bboxes,
                                                       det_labels,
                                                       self.test_cfg.rcnn,
                                                       ori_shape, scale_factor,
                                                       rescale, det_obj_ids=det_obj_ids)
        return segm_result

    def simple_video_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         mask_roi_map,
                         rescale=False):
        
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            obj_segms = []
            mask_clusters = [[]]
            mask_cluster_ids = []
            cluster_map = np.zeros((0, 0, 6), dtype=np.float32)
            roi_cluster_ids = [] 
            roi_indeces = []
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale:
                det_bboxes[:, :4] *= scale_factor
            mask_rois = bbox2roi([det_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            
            mask_cluster_pred = self.mask_cluster_head(mask_feats)
            
            obj_segms, mask_cluster_ids, roi_indeces, cluster_map = self.mask_cluster_head.get_cluster_mask_ids(
                                                    mask_cluster_pred,  det_bboxes,
                                                    det_labels,
                                                    self.test_cfg.rcnn,
                                                    ori_shape, scale_factor,
                                                    rescale, roi_id=False)

            mask_cluster_ids = [mask_cluster_id + 1 for mask_cluster_id in mask_cluster_ids]
            mask_cluster_pred = mask_cluster_pred[roi_indeces, mask_cluster_ids]
            mask_feats = mask_feats[roi_indeces]
            
            if len(roi_indeces) > 0:
                roi_cluster_pred = self.roi_cluster_head(mask_feats, mask_cluster_pred)
                roi_cluster_pred = roi_cluster_pred.cpu().numpy()
                roi_cluster_ids = np.argmax(roi_cluster_pred, axis=1)
                roi_cluster_ids = [pred[0]+1 for pred in roi_cluster_ids]
                if len(np.unique(np.asarray(roi_cluster_ids))) > 1:
                    mask_roi_map_tmp = {}
                    for m, r in zip(mask_cluster_ids, roi_cluster_ids):                        
                        mask_roi_map_tmp[m] = r

                    if len(list(mask_roi_map_tmp.values())) == self.num_clusters and len(np.unique(np.asarray(list(mask_roi_map_tmp.values())))) == self.num_clusters:
                        mask_roi_map = mask_roi_map_tmp
                else:
                    if mask_roi_map:
                        roi_cluster_ids = [mask_roi_map[m] for m in mask_cluster_ids]

                obj_segms = {roi_cluster_id:obj_segm for roi_cluster_id, obj_segm in zip(roi_cluster_ids, obj_segms)}
            else:
                roi_cluster_ids = []
        return obj_segms, cluster_map, roi_cluster_ids, roi_indeces, mask_roi_map

    def simple_test(self, img, img_meta, proposals=None, rescale=False, with_cluster=True):
        if not with_cluster:
            return super(ClusterRCNN, self).simple_test(img, img_meta, 
                                                            proposals=proposals, 
                                                            rescale=rescale)

        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        
        x = self.extract_feat(img)
        
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        
        segm_results, mask_clusters, mask_cluster_ids = self.simple_video_test_mask(
            x, img_meta, det_bboxes, det_labels, rescale=rescale)
        
        bbox_results = bbox2result_with_id(det_bboxes, det_labels, mask_cluster_ids, self.bbox_head.num_classes)
        
        return bbox_results, segm_results, mask_clusters

    def gen_colormask(self, N=256):
        def bitget(byteval, idx):
          return ((byteval & (1 << idx)) != 0)

        dtype = 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
          r = g = b = 0
          c = i
          for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

          cmap[i] = np.array([r, g, b])

        self.color_mask = cmap[3:]
    
    def show_result(self,
                    data,
                    result, dataset=None, score_thr=0.3):
        if isinstance(result, tuple):
            bbox_results, segm_results, mask_clusters, mask_clusters_orig = result
        else:
            bbox_results, segm_results, mask_clusters, mask_clusters_orig = result, None, None, None

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))
        # use colors to denote different object instances in videos
        # for YTVOS, only refresh color_mask at the first frame of each video
        if not hasattr(self, 'color_mask'):
          self.gen_colormask()
        if isinstance(bbox_results, dict) and len(bbox_results.keys()) == 0:
            return
                

        for inp, bbox_result, segm_result, mask_cluster, mask_cluster_orig in zip(data['inp_seq'], bbox_results, segm_results, mask_clusters, mask_clusters_orig):

            img_tensor = inp['img'][0]
            img_meta = inp['img_meta'][0].data[0][0]

            img = tensor2imgs(img_tensor, **img_meta['img_norm_cfg'])[0]

            h, w, _ = img_meta['img_shape']
            img = imresize(img, (w, h))
            #img_show = img[:h, :w, :]
            img_show = np.ones((h,w,3))

            if bbox_result:
                bboxes = np.vstack([x['bbox'] for x in bbox_result.values()])
                obj_ids = list(bbox_result.keys())
            else:
                bboxes = np.zeros((0, 5))
                obj_ids = []

            # draw segmentation masks

            if len(segm_result)>0:
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    if obj_ids[i] in segm_result:
                        mask = mask_util.decode(segm_result[obj_ids[i]]).astype(np.bool)
                        color_id = obj_ids[i] +1
                        img_show[mask] = self.color_mask[color_id,:]
            mask_cluster = (mask_cluster * 255).astype(np.uint8)
            mask_cluster_orig = (mask_cluster_orig * 255).astype(np.uint8)

            mask_cluster_img = np.concatenate((mask_cluster_orig[:, :, [0, 1, 2]], mask_cluster_orig[:, :, [0, 3, 4]]))
            #mask_cluster_img = np.concatenate((mask_cluster, mask_cluster_orig))
            save_vis = True
            save_path = 'viss'
            if save_vis:
              show = False 
              save_path = '{}/{}/{}.png'.format(save_path, img_meta['video_id'], img_meta['frame_id'])
            else:
              show = True
              save_path = None
            # draw bounding boxes
            if dataset == 'coco':
              labels = [
                  np.full(bbox.shape[0], i, dtype=np.int32)
                  for i, bbox in enumerate(bbox_result)
              ]
              labels = np.concatenate(labels)
            else:
              labels = [ x['label'] for x in bbox_result.values()]
              labels = np.array(labels)
            conc = np.concatenate((img_show, img))
            if mask_cluster_img.shape[0] == 0:
                mask_cluster_img = np.zeros_like(conc)
            mmcv.imshow_det_bboxes(                
                np.concatenate((conc, mask_cluster_img), axis=1),
                bboxes[:,:4],
                labels,
                class_names=class_names,
                show=show,
                text_color ='white',
                out_file=save_path)