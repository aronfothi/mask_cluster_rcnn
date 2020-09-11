import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from mmdet.core import bbox2roi, build_assigner, build_sampler, bbox2result_with_id
from .cluster_rcnn import ClusterRCNN
from .. import builder
from ..registry import DETECTORS

def chunker_list(seq, length, shift):
        return [seq[i:i+length] for i in range(len(seq))[0::shift]]

def segm_iou(trk, det):
  if np.any([trk, det], axis=0).sum() > 0:
    return np.all([trk, det], axis=0).sum() / np.any([trk, det], axis=0).sum()
  else:
    return 0

def matching(det_mask, trk_mask, ids):
    IOU_mat= np.zeros((len(ids),len(ids)),dtype=np.float32)
    for d,det_id in enumerate(ids):
        det = det_mask==det_id
        for t,trk_id in enumerate(ids):
            trk = trk_mask==trk_id    
            IOU_mat[d,t] = segm_iou(det, trk)

    return linear_sum_assignment(-IOU_mat)[1]

@DETECTORS.register_module
class ClusterRCRNN(ClusterRCNN):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 backbone_of=None,
                 shared_head=None,
                 mask_head= None,
                 mask_cluster_head=None,
                 roi_cluster_head=None,
                 gru_model=None,
                 pretrained=None,
                 swapped=False,
                 bidirectional=False,
                 num_clusters=2,
                 rnn_level=0):
        super(ClusterRCRNN, self).__init__(
            backbone=backbone,
            neck=neck,
            backbone_of=backbone_of,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            mask_cluster_head=mask_cluster_head,
            roi_cluster_head=roi_cluster_head,
            pretrained=pretrained,
            swapped=swapped,
            num_clusters=num_clusters)
        self.gru_model=builder.build_head(gru_model)
        self.bidirectional = bidirectional
        self.rnn_level = rnn_level
        self.num_clusters = num_clusters

        self.mask_roi_map = {}
        self.prev_mask_roi_map = {}

        self.prev_mask_clusters = None
        self.ids = np.arange(1, num_clusters+1)
        self.inv_mask = np.arange(num_clusters+1)
        
    def forward_train(self, img, img_meta, inp_seq, ref_frame_index, return_targets=False):

        M_seq = []

        if self.bidirectional:
            for inp in inp_seq:
                
                fms = list(self.backbone((inp['img'])))
                M_seq.append(fms)
            
            fms_f = []
            hidden = None
            for fms in M_seq:
                hidden = self.gru_model(fms[self.rnn_level], hidden)
                fms_f.append(hidden[-1])

            fms_b = []
            hidden = None
            for fms in reversed(M_seq):
                hidden = self.gru_model(fms[self.rnn_level], hidden)
                fms_b.append(hidden[-1])

            fms_b.reverse()

            for i, (fm_f, fm_b) in enumerate(zip(fms_f, fms_b)):
                M_seq[i][self.rnn_level] = torch.cat((fm_f, fm_b), 1)

                M_seq[i] = self.neck(M_seq[i])
        else:
            hidden = None
            for im in img_seq:
                fms = list(self.forward_to_neck(im))  

                hidden = self.gru_model(fms[self.rnn_level], hidden)
                fms[self.rnn_level] = hidden[-1]
                M_seq.append(fms)

         
        seq_targets = []
        for i, (M, inp) in enumerate(zip(M_seq, inp_seq)):
            calculate_losses = (i == ref_frame_index[0])

            l, targets = self.forward_heads( M,
                                                inp['img_meta'],
                                                inp['gt_bboxes'],
                                                inp['gt_labels'],
                                                inp.get('gt_inst_ids', None),  
                                                inp['gt_masks'], 
                                                return_targets=True,
                                                calculate_losses=calculate_losses,
                                                img=img)
            if calculate_losses:
                losses = l
            seq_targets.append(targets)

        if self.with_cluster:
            def batchseq2seqbatch(targetkey):
                seqbatchtarget=[target[targetkey] for target in seq_targets]
                seqbatchtarget=list(map(list, zip(*seqbatchtarget)))
                seqbatchtarget=[torch.cat(seq) for seq in seqbatchtarget]
                return seqbatchtarget

            def updated_assigned_gt_inds(target_key):
                assigned_gt_inds = []
                gt_bboxes_seq=[target['gt_bboxes'] for target in seq_targets]
                gt_bboxes_seq=list(map(list, zip(*gt_bboxes_seq))) # batch x seq

                assigned_gt_inds_seq=[target[target_key] for target in seq_targets] # seq x batch
                assigned_gt_inds_seq=list(map(list, zip(*assigned_gt_inds_seq))) # batch x seq
                
                            
                for ind_seq_batch, gt_bboxes_batch in zip(assigned_gt_inds_seq, gt_bboxes_seq):
                    ind_joint = []
                    shift = 0
                    for ind_seq, bboxes_seq in zip(ind_seq_batch, gt_bboxes_batch):                
                        ind_joint.append(ind_seq + shift)
                        shift += len(bboxes_seq)
                    assigned_gt_inds.append(torch.cat(ind_joint))

                return assigned_gt_inds

            pos_assigned_gt_inds = updated_assigned_gt_inds('pos_assigned_gt_inds')
            ext_assigned_gt_inds = updated_assigned_gt_inds('ext_assigned_gt_inds')
            gt_bboxes = batchseq2seqbatch('gt_bboxes')    
            mask_cluster_preds = batchseq2seqbatch('mask_cluster_pred')
            mask_inst_targets = batchseq2seqbatch('mask_inst_targets')
            mask_inst_score = batchseq2seqbatch('mask_inst_score')
            roi_cluster_preds = batchseq2seqbatch('roi_cluster_pred')            
            roi_inst_targets = batchseq2seqbatch('roi_inst_targets')
            roi_inst_score = batchseq2seqbatch('roi_inst_score')
            roi_bboxes = batchseq2seqbatch('roi_bboxes')        
            
            im_size_list= seq_targets[0]['im_size_list']

            roi_cluster_list = [self.train_cfg.rcnn.roi_cluster for _ in im_size_list]
            mask_pair_idx = self.mask_cluster_head.get_pairs(pos_assigned_gt_inds,
                                                                mask_inst_targets,
                                                                gt_bboxes,
                                                                im_size_list, self.train_cfg.rcnn)

            roi_pair_idx = self.mask_cluster_head.get_roi_pairs(ext_assigned_gt_inds,
                                                                gt_bboxes,
                                                                im_size_list, self.train_cfg.rcnn)

            mask_cluster_preds = torch.cat(mask_cluster_preds, 0)
            mask_inst_targets = torch.cat(mask_inst_targets, 0)
            mask_inst_score = torch.cat(mask_inst_score, 0)

            roi_cluster_preds = torch.cat(roi_cluster_preds, 0)
            roi_inst_targets = torch.cat(roi_inst_targets, 0)
            roi_inst_score = torch.cat(roi_inst_score, 0)

            roi_bboxes = torch.cat(roi_bboxes, 0)
            
            # DEBUG
            DEBUG = False
            if DEBUG:
                #bboxes = torch.cat([res.pos_bboxes for res in sampling_results])
                prop_nums = [[len(inds) for inds in target['pos_assigned_gt_inds']] for target in seq_targets]
                prop_nums=list(map(list, zip(*prop_nums)))
                prop_nums=sum(prop_nums, [])
                img_seq = [i['img'] for i in inp_seq]
                debug_img = list(map(list, zip(*img_seq)))
                debug_img = sum(debug_img, [])
                self.mask_cluster_head.display_instance_pairs(debug_img, prop_nums, mask_inst_targets, roi_inst_targets, roi_bboxes, mask_pair_idx, 0)
            # DEBUG END

            mask_cluster_preds = mask_cluster_preds[:, 1:]
            mask_cluster_preds = mask_cluster_preds.view(mask_cluster_preds.shape[0],
                                                         mask_cluster_preds.shape[1], -1)
            
            loss_mask_cluster = self.mask_cluster_head.clustering_loss(mask_cluster_preds,
                                                                mask_inst_targets,
                                                                mask_inst_score,
                                                                mask_pair_idx)
            losses.update(loss_mask_cluster) 

            loss_roi_cluster = self.roi_cluster_head.clustering_loss(roi_cluster_preds,
                                                                        roi_inst_targets,
                                                                        roi_inst_score,
                                                                        roi_pair_idx)
            losses.update(loss_roi_cluster)    
        
        if return_targets:
            return losses, seq_targets
        else:
            return losses


    def simple_test(self, 
                    img, 
                    img_meta, 
                    inp_seq=None,
                    rescale=False,
                    show=False):

        
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        bbox_result_seq = []
        segm_result_seq = []
        mask_cluster_seq = []
        mask_cluster_orig_seq = []

        

        
        M_seq = []

        if self.bidirectional:
            for inp in inp_seq:
                fms = list(self.backbone(inp['img'][0]))
                M_seq.append(fms)
            
            fms_f = []
            hidden = None
            for fms in M_seq:
                hidden = self.gru_model(fms[self.rnn_level], hidden)
                fms_f.append(hidden[-1])

            fms_b = []
            hidden = None
            for fms in reversed(M_seq):
                hidden = self.gru_model(fms[self.rnn_level], hidden)
                fms_b.append(hidden[-1])

            fms_b.reverse()

            for i, (fm_f, fm_b) in enumerate(zip(fms_f, fms_b)):
                M_seq[i][self.rnn_level] = torch.cat((fm_f, fm_b), 1)

                M_seq[i] = self.neck(M_seq[i])
        else:
            hidden = None
            for im in img_seq:
                fms = list(self.forward_to_neck(im))  

                hidden = self.gru_model(fms[self.rnn_level], hidden)
                fms[self.rnn_level] = hidden[-1]
                M_seq.append(fms)

        #extract end
        
        for e, (x, inp) in enumerate(zip(M_seq, inp_seq)):
            img_meta = inp['img_meta'][0]
            proposal_list = self.simple_test_rpn(
                x, img_meta, self.test_cfg.rpn)
            
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
            
            segm_results, mask_clusters_orig, roi_cluster_ids, roi_indeces, self.mask_roi_map = self.simple_video_test_mask(
                x, img_meta, det_bboxes, det_labels, self.mask_roi_map, rescale=rescale)

            mask_clusters_o = np.concatenate((mask_clusters_orig, np.zeros_like(mask_clusters_orig[:, :, :1])), axis=2)
            mask_clusters_o = mask_clusters_o.argmax(2)
            
            if e==0:
                if self.prev_mask_clusters is not None:
                    matched_idx = matching(mask_clusters_o, self.prev_mask_clusters, self.ids)
                    self.inv_mask = np.concatenate([np.zeros(1), self.ids[matched_idx]])

            mask_clusters = self.inv_mask[mask_clusters_o]

            def inv_roi(orig_roi_id):
                key_list = list(self.mask_roi_map.keys()) 
                val_list = list(self.mask_roi_map.values()) 
                orig_mask_id = key_list[val_list.index(orig_roi_id)]
                mask_id = self.inv_mask[orig_mask_id]
                return self.prev_mask_roi_map[mask_id]

            if self.prev_mask_roi_map:
                if isinstance(segm_results,dict):
                    segm_results = {self.inv_mask[roi_cluster_id]:obj_segm for roi_cluster_id, obj_segm in segm_results.items()}
                roi_cluster_ids = [inv_roi(roi_cluster_id) for roi_cluster_id in roi_cluster_ids]
            else:
                self.prev_mask_roi_map = self.mask_roi_map

            det_bboxes = det_bboxes[roi_indeces]
            det_labels = det_labels[roi_indeces]
            
            bbox_results = bbox2result_with_id(det_bboxes, det_labels, roi_cluster_ids, self.bbox_head.num_classes)

            
            bbox_result_seq.append(bbox_results)
            segm_result_seq.append(segm_results)
            mask_clusters_vis = np.concatenate(((mask_clusters==1)[:, :, None], (mask_clusters==2)[:, :, None], (mask_clusters==3)[:, :, None]), axis=2)
            mask_cluster_seq.append(mask_clusters_vis)
            mask_cluster_orig_seq.append(mask_clusters_orig)

            if e==len(M_seq)-1:
                self.prev_mask_clusters = mask_clusters
                self.prev_mask_roi_map = self.mask_roi_map

        
        if show:
            return bbox_result_seq, segm_result_seq, mask_cluster_seq, mask_cluster_orig_seq
        else:
            return bbox_result_seq, segm_result_seq
