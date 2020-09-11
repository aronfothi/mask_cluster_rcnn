import numpy as np
import torch
import torch.nn as nn
from mmcv import imresize
from mmcv.cnn import kaiming_init, normal_init
from mmdet.core import force_fp32, auto_fp16

from ..builder import build_loss
from ..registry import HEADS

from ..utils import ConvModule
from mmdet.core.utils import multi_apply

from mmdet.core import mask_target

import matplotlib.pyplot as plt
import pycocotools.mask as mask_util

import random

def random_colors(n):
  ret = []
  r = int(random.random() * 256)
  g = int(random.random() * 256)
  b = int(random.random() * 256)
  step = 256 / n
  for i in range(n):
    r += step
    g += step
    b += step
    r = int(r) % 256
    g = int(g) % 256
    b = int(b) % 256
    ret.append((r,g,b)) 
  return ret

def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


@HEADS.register_module
class MaskClusterHead(nn.Module):
    """Mask Cluster Head.

    This head predicts the clusters.
    """

    def __init__(self,
                 num_convs=4,
                 num_fcs=2,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 conv_cfg=None,
                 norm_cfg=None,
                 num_instances=5*5,
                 max_distance=1.0,
                 num_samples=100,
                 num_clusters=5+1,
                 roi_cluster=False,
                 loss_cluster=None,
                 loss_background=None):
        super(MaskClusterHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.fc_out_channels = fc_out_channels
        self.num_clusters = num_clusters
        self.num_instances = num_instances
        self.max_distance = max_distance
        self.num_samples = num_samples
        self.roi_cluster = roi_cluster

        self.loss_cluster = build_loss(loss_cluster)
        if not self.roi_cluster:
            self.loss_background = build_loss(loss_background)

        self.convs = nn.ModuleList()
        for i in range(num_convs):
            if i == 0:
                # concatenation of mask feature and mask prediction
                in_channels = self.in_channels
            else:
                in_channels = self.conv_out_channels
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        

        if self.roi_cluster:
            roi_feat_size = torch.nn.modules.utils._pair(roi_feat_size)
            #pooled_area = (roi_feat_size[0] // 2) * (roi_feat_size[1] // 2)
            area = (roi_feat_size[0]) * (roi_feat_size[1])
            self.fcs = nn.ModuleList()
            for i in range(num_fcs):            

                in_channels = (
                    self.conv_out_channels *
                    area if i == 0 else self.fc_out_channels)
                self.fcs.append(nn.Linear(in_channels, self.fc_out_channels))
                
            self.fc_roi_cluster = nn.Linear(self.fc_out_channels, self.num_clusters)
            self.max_pool = nn.MaxPool2d(2, 2)

        else:
            upsample_in_channels = (
                self.conv_out_channels if self.num_convs > 0 else in_channels)
            if self.upsample_method is None:
                self.upsample = None
            elif self.upsample_method == 'deconv':
                self.upsample = nn.ConvTranspose2d(
                    upsample_in_channels,
                    self.conv_out_channels,
                    self.upsample_ratio,
                    stride=self.upsample_ratio)
            else:
                self.upsample = nn.Upsample(
                    scale_factor=self.upsample_ratio, mode=self.upsample_method)

            out_channels = self.num_clusters
            logits_in_channel = (
                self.conv_out_channels
                if self.upsample_method == 'deconv' else upsample_in_channels)

            self.conv_logits = nn.Conv2d(logits_in_channel, out_channels, 1)
        
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

        self.inst_colors = np.vstack([(0, 0, 0)] + random_colors(num_clusters-1))
        

    def init_weights(self):
        if self.roi_cluster:

            for fc in self.fcs:
                kaiming_init(
                    fc,
                    a=1,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    distribution='uniform')

            normal_init(self.fc_roi_cluster, std=0.01)
        else:
            for m in [self.upsample, self.conv_logits]:
                if m is None:
                    continue
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    
    @auto_fp16()
    def forward(self, x, mask_pred=None): 

        if self.roi_cluster:
            mask_pred_pooled = self.max_pool(mask_pred.unsqueeze(1))

            x = torch.cat((x, mask_pred_pooled), 1)

        for conv in self.convs:
            x = conv(x)

        if self.roi_cluster:
            c = x.view(x.size(0), -1)
            for fc in self.fcs:
                c = self.relu(fc(c))
            roi_cluster_pred = self.fc_roi_cluster(c)
            return roi_cluster_pred.softmax(1).unsqueeze(-1)
        
        else:
            if self.upsample is not None:
                x = self.upsample(x)
                if self.upsample_method == 'deconv':
                    x = self.relu(x)
            mask_cluster_pred = self.conv_logits(x)            
            
            return mask_cluster_pred.softmax(1)


    @force_fp32(apply_to=('pred'))
    def clustering_loss(self, pred, target, score, pair):
        if self.roi_cluster:
            return dict(loss_roi_cluster=self.loss_cluster(pred, target, score, pair))
        else:
            return dict(loss_mask_cluster=self.loss_cluster(pred, target, score, pair))

    

    def display_targets(self, target, img, mask_target, inst_target, mask_cluster_pred):
        img_np = img[0].cpu().numpy()
        img_np_2 = img[1].cpu().numpy()
        targ = (target.cpu().numpy()*255).astype(np.uint8)
        mask_targ = (mask_target.cpu().numpy()*255).astype(np.uint8)
        inst_targ = (inst_target.cpu().numpy())
        mask_cluster = (mask_cluster_pred.cpu().detach().numpy()*255).astype(np.uint8)

        img_np = (img_np.transpose((1, 2, 0)) + np.min(img_np)) /(np.max(img_np)-np.min(img_np)) *255
        img_np_2 = (img_np_2.transpose((1, 2, 0)) + np.min(img_np_2)) /(np.max(img_np_2)-np.min(img_np_2)) *255
        
        display_images([img_np.astype(np.uint8), img_np_2.astype(np.uint8) ] +  list(targ)[:3] + list(mask_targ)[:3] + list(inst_targ)[:3] + list(mask_cluster)[:3])
        
    def display_instances(self, img, inst_targets, bboxes, softmaxes):
        img_np = img.cpu().numpy()
        img_np = (img_np.transpose((1, 2, 0)) - np.min(img_np)) /(np.max(img_np)-np.min(img_np))
        img_np *= 255      
        inst_targets_np = (inst_targets.cpu().numpy()).astype(np.uint8)
        softmaxes_np = softmaxes.detach().cpu().numpy().transpose(0, 2, 3, 1)
        bboxes_np = (bboxes.cpu().numpy()).astype(np.uint16)
        colors = random_colors(len(np.unique(inst_targets_np)))
        colors = np.vstack(colors)
        res = []
        for bbox, inst_target, sm in zip(bboxes_np, inst_targets_np, softmaxes_np):
            im = img_np.copy()
            im_sm1 = img_np.copy()
            im_sm2 = img_np.copy()
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            roi_im = im[y1:y1 + h, x1:x1 + w, :]
            
            roi_im_res = imresize(roi_im, (28, 28))
            whr_x, whr_y = np.where(inst_target>0)

            roi_im_res[whr_x, whr_y] = colors[inst_target[whr_x, whr_y]-1]
            
            roi_im = imresize(roi_im_res, (roi_im.shape[1], roi_im.shape[0]))
            im[y1:y1 + h, x1:x1 + w, :] = roi_im

            roi_sm1 = imresize(sm[:, :, :3], (roi_im.shape[1], roi_im.shape[0]))
            roi_sm2 = imresize(sm[:, :, 3:], (roi_im.shape[1], roi_im.shape[0]))

            im_sm1[y1:y1 + h, x1:x1 + w, :] = (roi_sm1 * 255).astype(np.uint8)
            im_sm2[y1:y1 + h, x1:x1 + w, :] = (roi_sm2 * 255).astype(np.uint8)

            res.append(im)
            res.append(im_sm1)
            res.append(im_sm2)

        display_images(res[:9], cols=3)

    def display_instance_pairs_x(self, img_list, inst_targets, pairs, mask_cluster_preds, inst_idx):
        res = []
        inst_targets_np = (inst_targets.cpu().numpy()).astype(np.uint8)
        mask_cluster_preds_np = (mask_cluster_preds.detach().cpu().numpy()).transpose(0, 2, 3, 1)
        
        pairs_np = (pairs.cpu().numpy()).astype(np.uint32)
        

        u, inv_inst_targets_np = np.unique(inst_targets_np, return_inverse=True)
        inv_inst_targets_np = inv_inst_targets_np.reshape(inst_targets_np.shape)
        colors = random_colors(len(u))
        colors = np.vstack(colors)
        
        inst_idx = pairs_np[:, 0][-1]
        pairs_np = pairs_np[pairs_np[:, 0]==inst_idx]
        first_pair = pairs_np[0]
        inst_id = inst_targets_np[first_pair[0], first_pair[1]] 

        print('inst_id', inst_id)

        for i, img in enumerate(img_list):
            inst_targets_np_img = inst_targets_np[i]
            mask_cluster_preds_np_img = mask_cluster_preds_np[i]
            inv_inst_targets_np_img = inv_inst_targets_np[i]
            mask_size = (mask_cluster_preds_np_img.shape[1], mask_cluster_preds_np_img.shape[0])

            img_np = img.cpu().numpy()
            img_np = (img_np.transpose((1, 2, 0)) - np.min(img_np)) /(np.max(img_np)-np.min(img_np))
            img_np *= 255      

            img_np_o = imresize(img_np, mask_size)
            img_np = imresize(img_np, mask_size).reshape(-1, 3)
            
            
            whr_x = np.where(inst_targets_np_img>0)[0]
 
            img_np[whr_x] = colors[inv_inst_targets_np_img[whr_x]-1]



            roi_pairs = pairs_np[pairs_np[:, 2]==i]
            img_np[roi_pairs[:, 3]] = [255, 0, 0] # red # if inst_targets_np[roi_pairs[:, 3], roi_pairs[:, 4], roi_pairs[:, 5]] else [0, 0, 255]
            R = inst_targets_np[roi_pairs[:, 2], roi_pairs[:, 3]] == inst_id
            pos_roi_pairs = roi_pairs[R]
            
            img_np[pos_roi_pairs[:, 3]] = [0, 0, 255] #blue
            
            img_np = img_np.reshape((mask_size[1], mask_size[0], 3))
            #roi_im = imresize(roi_im_res, (roi_im.shape[1], roi_im.shape[0]))
            #img_np[y1:y1 + h, x1:x1 + w, :] = roi_im
            res.append(img_np)
            res.append(img_np_o)

            im1 = (mask_cluster_preds_np_img* 255).astype(np.uint8)
            im2 = (mask_cluster_preds_np_img* 255).astype(np.uint8)
            res.append(im1)
            res.append(im2)

        display_images(res, cols=4)

    def display_instance_pairs(self, img_list, prop_num_list, inst_targets, roi_inst_targets, bbox_list, pairs, inst_idx):
        res = []
        inst_targets_np = (inst_targets.cpu().numpy()).astype(np.uint8)
        roi_inst_targets_np = (roi_inst_targets.cpu().numpy()).astype(np.uint8)

        
        pairs_np = (pairs.cpu().numpy()).astype(np.uint16)

        u, inv_inst_targets_np = np.unique(inst_targets_np, return_inverse=True)
        inv_inst_targets_np = inv_inst_targets_np.reshape(inst_targets_np.shape)
        colors = random_colors(len(u))
        colors = np.vstack(colors)
        #R = np.equal(inst_targets_np[pairs_np[:, 0], pairs_np[:, 1]], inst_targets_np[pairs_np[:, 2], pairs_np[:, 3], pairs_np[:, 5]])
        
        inst_idx = pairs_np[:, 0][-1]
        pairs_np = pairs_np[pairs_np[:, 0]==inst_idx]
        first_pair = pairs_np[0]
        inst_id = inst_targets_np[first_pair[0], first_pair[1]]   
        start_index = 0
        for img, prop_num in zip(img_list, prop_num_list):
            img_np = img.cpu().numpy()
            img_np = (img_np.transpose((1, 2, 0)) - np.min(img_np)) /(np.max(img_np)-np.min(img_np))
            img_np *= 255      
            img_np_o = img_np.copy()

            
            bboxes = bbox_list[start_index:start_index+prop_num]
            bboxes_np = (bboxes.cpu().numpy()).astype(np.uint16)

            inst_targets_np_img = inst_targets_np[start_index:start_index+bboxes_np.shape[0]]
            inv_inst_targets_np_img = inv_inst_targets_np[start_index:start_index+bboxes_np.shape[0]]

            roi_inst_targets_np_img = roi_inst_targets_np[start_index:start_index+bboxes_np.shape[0]]

            if start_index == 0:
                if len(np.unique(inst_targets_np_img)) < 3:
                    return
            
            assert inst_targets_np_img.shape[0] == bboxes_np.shape[0] == inv_inst_targets_np_img.shape[0]
            
            for bbox, inst_target, roi_inst_target in zip(bboxes_np, inv_inst_targets_np_img, roi_inst_targets_np_img):
                x1, y1, x2, y2 = bbox
                w = np.maximum(x2 - x1 + 1, 1)
                h = np.maximum(y2 - y1 + 1, 1)
                roi_im = img_np[y1:y1 + h, x1:x1 + w, :]
                
                roi_im_res = imresize(roi_im, (28, 28)).reshape(-1, 3)
                whr_x = np.where(inst_target>0)[0]

                assert len(whr_x) >= 5:
                roi_im_res[whr_x] = colors[inst_target[whr_x]-1]
                roi_im_res = roi_im_res.reshape((28, 28, 3))
                roi_im = imresize(roi_im_res, (roi_im.shape[1], roi_im.shape[0]))
                img_np[y1:y1 + h, x1:x1 + w, :] = roi_im

            
            for i, (bbox, inst_target) in enumerate(zip(bboxes_np, inst_targets_np_img)):
                ii = i+start_index
                x1, y1, x2, y2 = bbox
                w = np.maximum(x2 - x1 + 1, 1)
                h = np.maximum(y2 - y1 + 1, 1)
                roi_im = img_np[y1:y1 + h, x1:x1 + w, :]                
                roi_im_res = imresize(roi_im, (28, 28)).reshape(-1, 3)              

                roi_pairs = pairs_np[pairs_np[:, 2]==ii]

                roi_im_res[roi_pairs[:, 3]] = [255, 0, 0] # red # if inst_targets_np[roi_pairs[:, 3], roi_pairs[:, 4], roi_pairs[:, 5]] else [0, 0, 255]
                R = inst_targets_np[roi_pairs[:, 2], roi_pairs[:, 3]] == inst_id
                pos_roi_pairs = roi_pairs[R]
                
                roi_im_res[pos_roi_pairs[:, 3]] = [0, 0, 255] #blue
                roi_im_res = roi_im_res.reshape((28, 28, 3))
                roi_im = imresize(roi_im_res, (roi_im.shape[1], roi_im.shape[0]))
                img_np[y1:y1 + h, x1:x1 + w, :] = roi_im
                   
            start_index += prop_num
            res.append(img_np)
            res.append(img_np_o)

        display_images(res, cols=4)
        

    @force_fp32(apply_to=('mask_cluster_pred_1', 'mask_cluster_pred_2', ))
    def background_loss(self, pred, target, labels):
        loss = self.loss_background(pred, target, torch.zeros_like(labels))
        return dict(loss_background = loss)

    def _resize_inst_mask(self, inst_mask, mask_size):        
        inst_mask = imresize(inst_mask, mask_size)
        return inst_mask

    def _one_mask_target_single(self, gt_masks, inst_ids, mask_size):
        
        inst_ids_nd = inst_ids.cpu().numpy().astype(np.int32)               
        

        roi_masks = gt_masks.astype(np.float32)
        
        valid = np.any(roi_masks.reshape((roi_masks.shape[0], -1)), axis=1)
        roi_masks = roi_masks[valid]
        mask_roi_inst_ids = inst_ids_nd[valid]
        inst_number = inst_ids_nd.shape[0]

        bg_mask = np.equal(np.any(roi_masks, axis=0), np.zeros((roi_masks.shape[1], roi_masks.shape[2]))).astype(np.float32) 
        bg_target = imresize(bg_mask, mask_size)
        
        size_list = [mask_size for _ in range(valid.sum())]

        mask_target = map(self._resize_inst_mask, (roi_masks).astype(np.float32) , size_list)
                        
        mask_target = list(mask_target)
        #First should be zero
        mask_roi_inst_ids = np.concatenate((np.zeros(1, dtype=np.uint16), mask_roi_inst_ids))                
        mask_target = [np.zeros([mask_size[1], mask_size[0]], dtype=np.float64)] + mask_target
        mask_target = np.stack(mask_target)
    
        score_map = np.max(mask_target, axis=0)
        score_map = score_map * (1 + (inst_number/10))

        su = np.sum(mask_target, axis=0)
        mask_target[:, su>1.1] = 0
        mask_target = np.argmax(mask_target, axis=0)  
    
        mask_target = mask_roi_inst_ids[mask_target]
        mask_target = mask_target.flatten()
        score_map = score_map.flatten()
        
        bg_mask_targets = torch.from_numpy(np.stack([bg_target])).float().to(
            inst_ids.device)
        inst_mask_targets_tensor = torch.from_numpy(np.stack([mask_target])).float().to(
            inst_ids.device)

        return bg_mask_targets, inst_mask_targets_tensor

    def get_one_target(self, gt_masks, inst_ids, mask_sizes):
        """Compute target of mask cluster.

        M
        """       

        bg_mask, inst_mask = multi_apply(self._one_mask_target_single, 
                                        gt_masks, inst_ids, mask_sizes)
        bg_mask_targets = torch.cat(list(bg_mask))
        inst_mask_targets = inst_mask        
        
        return bg_mask_targets, inst_mask_targets

    def _mask_target_single(self, pos_proposals, pos_assigned_gt_inds, gt_masks, inst_ids, cfg):
        mask_size = cfg.mask_size
        num_pos = pos_proposals.size(0)
        bg_mask_targets = []
        inst_mask_targets = []
        inst_mask_scores = []

        inst_roi_targets = []
        inst_roi_scores = []
        
        if num_pos > 0:

            pos_proposals_nd = pos_proposals.cpu().numpy().astype(np.int32)
            inst_ids_nd = inst_ids.cpu().numpy().astype(np.int32)
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

            roi_inst_ids = inst_ids_nd[pos_assigned_gt_inds]

            #inst_sizes = gt_masks.reshape(gt_masks.shape[0], -1).sum(axis=1).argsort()
            #gt_masks = gt_masks[inst_sizes]
            #inst_ids_nd = inst_ids_nd[inst_sizes]
            
            for i in range(num_pos):
                bbox = pos_proposals_nd[i, :]
                x1, y1, x2, y2 = bbox
                w = np.maximum(x2 - x1 + 1, 1)
                h = np.maximum(y2 - y1 + 1, 1)
                # mask is uint8 both before and after resizing

                roi_masks = gt_masks[:, y1:y1 + h, x1:x1 + w].astype(np.float32)
                
                valid = np.any(roi_masks.reshape((roi_masks.shape[0], -1)), axis=1)
                roi_masks = roi_masks[valid]
                mask_roi_inst_ids = inst_ids_nd[valid]
                inst_number = inst_ids_nd.shape[0]

                bg_mask = np.equal(np.any(roi_masks, axis=0), np.zeros((roi_masks.shape[1], roi_masks.shape[2]))).astype(np.float32) 
                bg_target = imresize(bg_mask,
                                    (mask_size, mask_size))
                bg_mask_targets.append(bg_target)
                                
                size_list = [(mask_size, mask_size) for _ in range(valid.sum())]

                mask_target = map(self._resize_inst_mask, (roi_masks).astype(np.float32) , size_list)
                                
                mask_target = list(mask_target)
                #First should be zero
                mask_roi_inst_ids = np.concatenate((np.zeros(1, dtype=np.uint16), mask_roi_inst_ids))                
                mask_target = [np.zeros([mask_size, mask_size], dtype=np.float64)] + mask_target
                mask_target = np.stack(mask_target)
            
                score_map = np.max(mask_target, axis=0)
                score_map = score_map * (1 + (inst_number/10))

                su = np.sum(mask_target, axis=0)
                mask_target[:, su>1.1] = 0
                mask_target = np.argmax(mask_target, axis=0)  
            
                mask_target = mask_roi_inst_ids[mask_target]
                mask_target = mask_target.flatten()
                score_map = score_map.flatten()

                roi_inst_id = roi_inst_ids[i]
                inst_roi_targets.append(np.array([roi_inst_id]))
                inst_roi_scores.append(np.array([cfg.roi_cluster_score]))

                inst_mask_targets.append(mask_target)
                inst_mask_scores.append(score_map)
            
            bg_mask_targets = torch.from_numpy(np.stack(bg_mask_targets)).float().to(
                pos_proposals.device)
            inst_mask_targets_tensor = torch.from_numpy(np.stack(inst_mask_targets)).float().to(
                pos_proposals.device)
            inst_mask_scores_tensor = torch.from_numpy(np.stack(inst_mask_scores)).float().to(
                pos_proposals.device)  

            inst_roi_targets_tensor = torch.from_numpy(np.stack(inst_roi_targets)).float().to(
                pos_proposals.device)
            inst_roi_scores_tensor = torch.from_numpy(np.stack(inst_roi_scores)).float().to(
                pos_proposals.device)
                         
        else:            
            bg_mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
            inst_mask_targets_tensor = pos_proposals.new_zeros((0, mask_size * mask_size ))
            inst_mask_scores_tensor = pos_proposals.new_zeros((0, mask_size * mask_size))

            inst_roi_targets_tensor = pos_proposals.new_zeros((0, 1))
            inst_roi_scores_tensor = pos_proposals.new_zeros((0, 1))

        return bg_mask_targets, inst_mask_targets_tensor, inst_mask_scores_tensor, inst_roi_targets_tensor, inst_roi_scores_tensor

    def get_target(self, sampling_results, gt_masks, inst_ids,
                   rcnn_train_cfg):
        """Compute target of mask cluster.

        Mask IoU target is the IoU of the predicted mask (inside a bbox) and
        the gt mask of corresponding gt mask (the whole instance).
        The intersection area is computed inside the bbox, and the gt mask area
        is computed with two steps, firstly we compute the gt area inside the
        bbox, then divide it by the area ratio of gt area inside the bbox and
        the gt area of the whole instance.

        Args:
            sampling_results (list[:obj:`SamplingResult`]): sampling results.
            gt_masks (list[ndarray]): Gt masks (the whole instance) of each
                image, binary maps with the same shape of the input image.
            mask_targets (Tensor): Gt mask of each positive proposal,
                binary map of the shape (num_pos, h, w).
            rcnn_train_cfg (dict): Training config for R-CNN part.

        Returns:
            Tensor: mask iou target (length == num positive).
        """       

        pos_proposals = [res.pos_bboxes for res in sampling_results] 
        pos_assigned_gt_ind_list = [res.pos_assigned_gt_inds for res in sampling_results]

        cfg_list = [rcnn_train_cfg for _ in range(len(pos_proposals))]
        bg_mask, inst_mask, inst_mask_scores, inst_roi, inst_roi_scores = multi_apply(self._mask_target_single, 
                                                            pos_proposals, pos_assigned_gt_ind_list,
                                                            gt_masks, inst_ids, cfg_list)
        bg_mask_targets = torch.cat(list(bg_mask))
        inst_mask_targets = inst_mask
        inst_mask_scores = inst_mask_scores
        inst_roi_targets = inst_roi
        inst_roi_scores = inst_roi_scores
        
        return bg_mask_targets, inst_mask_targets, inst_mask_scores, inst_roi_targets, inst_roi_scores

    def _roi_mask_target_single(self, pos_proposals, pos_assigned_gt_inds, gt_masks, inst_ids, mask_cluster_preds, cfg):
        mask_size = cfg.mask_size
        num_pos = pos_proposals.size(0)        

        inst_roi_targets = []
        roi_indeces = []
        cluster_indeces = []
        ext_assigned_gt_inds = []
        
        if num_pos > 0:

            pos_proposals_nd = pos_proposals.cpu().numpy().astype(np.int32)
            mask_cluster_preds_nd = mask_cluster_preds.detach().cpu().numpy()
            inst_ids_nd = inst_ids.cpu().numpy().astype(np.int32)
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

            roi_inst_ids = inst_ids_nd[pos_assigned_gt_inds]

            mask_sizes = gt_masks.sum(-1).sum(-1)
            

            for i in range(num_pos):
                roi_mask_cluster_preds = mask_cluster_preds_nd[i]
                roi_inst_id = roi_inst_ids[i]

                bbox = pos_proposals_nd[i, :]
                x1, y1, x2, y2 = bbox
                w = np.maximum(x2 - x1 + 1, 1)
                h = np.maximum(y2 - y1 + 1, 1)
                # mask is uint8 both before and after resizing


                assigned_gt_ind = pos_assigned_gt_inds[i]

                roi_masks = gt_masks[:, y1:y1 + h, x1:x1 + w].astype(np.float32)
                roi_mask_sizes = roi_masks.sum(-1).sum(-1)
                mask_in_roi = roi_mask_sizes / mask_sizes
                
                other_mask_in_roi_inds = np.where(mask_in_roi>.5)[0]
                other_mask_in_roi_inds = other_mask_in_roi_inds[other_mask_in_roi_inds!=assigned_gt_ind]
                other_mask_in_roi_ids = inst_ids_nd[other_mask_in_roi_inds]
                
                valid = np.any(roi_masks.reshape((roi_masks.shape[0], -1)), axis=1)
                roi_masks = roi_masks[valid]
                mask_roi_inst_ids = inst_ids_nd[valid]
                inst_number = inst_ids_nd.shape[0]
                                
                size_list = [(mask_size, mask_size) for _ in range(valid.sum())]

                mask_target = map(self._resize_inst_mask, (roi_masks).astype(np.float32) , size_list)
                                
                mask_target = list(mask_target)
                #First should be zero
                
                mask_roi_inst_ids = np.concatenate((np.zeros(1, dtype=np.uint16), mask_roi_inst_ids))                
                mask_target = [np.zeros([mask_size, mask_size], dtype=np.float64)] + mask_target
                mask_target = np.stack(mask_target)
            
                score_map = np.max(mask_target, axis=0)
                score_map = score_map * (1 + (inst_number/10))
                mask_target = np.argmax(mask_target, axis=0)
            
                mask_target = mask_roi_inst_ids[mask_target]
                
                for c in np.concatenate(([roi_inst_id], other_mask_in_roi_ids)):
                    sum_map = roi_mask_cluster_preds.transpose(1, 2, 0)[mask_target==roi_inst_id]
                    sums = np.sum(sum_map, axis=0)

                    choosen_cluster = np.argmax(sums[1:]) + 1

                    inst_roi_targets.append(np.array([roi_inst_id]))
                    roi_indeces.append(i)
                    cluster_indeces.append(choosen_cluster)
                    ext_assigned_gt_inds.append(assigned_gt_ind)

            inst_roi_targets_tensor = torch.from_numpy(np.stack(inst_roi_targets)).float().to(
                pos_proposals.device)
            roi_indeces_tensor = torch.from_numpy(np.stack(roi_indeces)).to(
                pos_proposals.device)
            cluster_indeces_tensor = torch.from_numpy(np.stack(cluster_indeces)).to(
                pos_proposals.device)
            ext_assigned_gt_inds_tensor = torch.from_numpy(np.stack(ext_assigned_gt_inds)).to(
                pos_proposals.device)
                         
        else:            
            inst_roi_targets_tensor = pos_proposals.new_zeros((0, 1))
            roi_indeces_tensor = pos_proposals.new_zeros((0, 1))
            cluster_indeces_tensor = pos_proposals.new_zeros((0, 1))
            ext_assigned_gt_inds_tensor = pos_proposals.new_zeros((0, 1))

        return inst_roi_targets_tensor, roi_indeces_tensor, cluster_indeces_tensor, ext_assigned_gt_inds_tensor


    def get_roi_mask_target(self, sampling_results, gt_masks, inst_ids, mask_cluster_preds, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results] 
        pos_assigned_gt_ind_list = [res.pos_assigned_gt_inds for res in sampling_results]


        cfg_list = [rcnn_train_cfg for _ in range(len(pos_proposals))] 
        inst_roi, roi_indeces, cluster_indeces, assigned_gt_inds = multi_apply(self._roi_mask_target_single, 
                                                            pos_proposals, pos_assigned_gt_ind_list,
                                                            gt_masks, inst_ids, mask_cluster_preds, cfg_list)
        
        return inst_roi, roi_indeces, cluster_indeces, assigned_gt_inds

    def _roi_target_single(self, pos_assigned_gt_inds, inst_ids):
        
        return inst_ids[pos_assigned_gt_inds].view(-1, 1)

    def get_roi_target(self, pos_assigned_gt_inds, inst_ids):
        inst_roi = map(self._roi_target_single, pos_assigned_gt_inds, inst_ids)
        return inst_roi
    
    def _roi_distance(self, boxes1, boxes2, im_size, max_num_instances=100):
        """Computes distance between two sets of boxes.
        boxes1, boxes2: [N, (y1, x1, y2, x2)].
        """
        # 1. Tile boxes2 and repeat boxes1. This allows us to compare
        # every boxes1 against every boxes2 without loops.        
    
        b1_size = boxes1.size()
        b2_size = boxes2.size()

        b1 = boxes1.repeat(1, b2_size[0]).view(-1, 4)
        b2 = boxes2.repeat(b1_size[0], 1)        
        
        b1_y1, b1_x1, b1_y2, b1_x2 = torch.split(b1, 1, dim=-1)
        b2_y1, b2_x1, b2_y2, b2_x2 = torch.split(b2, 1, dim=-1)

        y1 = torch.max(b1_y1, b2_y1)
        x1 = torch.max(b1_x1, b2_x1)
        y2 = torch.min(b1_y2, b2_y2)
        x2 = torch.min(b1_x2, b2_x2)

        dy = torch.max(y1 - y2, torch.zeros_like(y1))
        dx = torch.max(x1 - x2, torch.zeros_like(x1))

        dqrt = dx * dx + dy * dy

        dqrt = dqrt.view(b1_size[0], b2_size[0])

        distances = torch.sqrt(dqrt.float())
        distances = distances / im_size

        bbox_rank_0 = torch.argsort(torch.argsort(distances, dim=0), dim=0)
        bbox_rank_1 = torch.argsort(torch.argsort(distances, dim=1), dim=1)

        xtr_dist = torch.ones_like(distances) *10

        distances = torch.where(bbox_rank_0 <= (max_num_instances-1), distances, xtr_dist)
        distances = torch.where(bbox_rank_1 <= (max_num_instances-1), distances, xtr_dist)
        return  distances

    def _sample_instances(self, inst_mask_targets, num_samples, roi_cluster):
        sampled_idx = np.ones((inst_mask_targets.shape[0], num_samples, 2), dtype=int) * -1
        for i in range(inst_mask_targets.shape[0]):  
            whr_x = np.where(inst_mask_targets[i]>0)[0]
            if(len(whr_x)>0):
                sampled_perm = np.random.permutation(len(whr_x))
                sampled_perm = sampled_perm[:num_samples]
                whr_x = whr_x[sampled_perm]
                
                sampled_idx[i, :len(whr_x), 0] = i
                sampled_idx[i, :len(whr_x), 1] = whr_x

        return sampled_idx

    def _get_pairs_single(self, pos_assigned_gt_ind,                        
                        inst_mask_targets,
                        gt_bboxes,
                        im_size, roi_cluster):

        sample = self._sample_instances(inst_mask_targets.cpu().numpy(), self.num_samples, roi_cluster)

        bbox_distance = self._roi_distance(gt_bboxes, gt_bboxes, im_size, self.num_instances)
        bbox_distance = bbox_distance.cpu().numpy()
        pos_assigned_gt_ind = pos_assigned_gt_ind.cpu().numpy()

        roi_distance = bbox_distance[pos_assigned_gt_ind, :][:, pos_assigned_gt_ind] #TODO can be swapped

        roi_idx_1, roi_idx_2 = np.where(roi_distance<=self.max_distance) 

        mask_pair_idx = np.hstack([sample[roi_idx_1].reshape(-1, 2), sample[roi_idx_2].reshape(-1, 2)])
        mask_pair_idx = mask_pair_idx[mask_pair_idx[:, 1]>-1]
        mask_pair_idx = mask_pair_idx[mask_pair_idx[:, 3]>-1]
        mask_pair_idx = torch.from_numpy(mask_pair_idx).long().to(gt_bboxes.device)
        return mask_pair_idx



    def get_pairs(self, pos_assigned_gt_inds,
                    mask_inst_targets,
                    gt_bboxes,
                    im_size_list, roi_cluster_list):
        
        mask_pair_idx = map(self._get_pairs_single,
                            pos_assigned_gt_inds,
                            mask_inst_targets,
                            gt_bboxes,
                            im_size_list, roi_cluster_list)

        mask_pair_idx = list(mask_pair_idx)
        shift = 0
        for mask_pair, prop in zip(mask_pair_idx, pos_assigned_gt_inds):

            mask_pair[:, 0] += shift
            mask_pair[:, 2] += shift
            shift += len(prop)
        mask_pair_idx = torch.cat(mask_pair_idx)
        return mask_pair_idx

    def _get_roi_pairs_single(self, ext_assigned_gt_ind,
                        gt_bboxes,
                        im_size, cfg):

        bbox_distance = self._roi_distance(gt_bboxes, gt_bboxes, im_size, self.num_instances)
        bbox_distance = bbox_distance.cpu().numpy()
        ext_assigned_gt_ind = ext_assigned_gt_ind.cpu().numpy()
        roi_distance = bbox_distance[ext_assigned_gt_ind, :][:, ext_assigned_gt_ind] #TODO can be swapped

        roi_idx_1, roi_idx_2 = np.where(roi_distance<=self.max_distance) 

        roi_pair_idx = np.vstack([roi_idx_1, np.zeros_like(roi_idx_1), roi_idx_2, np.zeros_like(roi_idx_2)])
        roi_pair_idx = torch.t(torch.from_numpy(roi_pair_idx).long().to(gt_bboxes.device))
        
        return roi_pair_idx

    def get_roi_pairs(self, ext_assigned_gt_inds,
                            gt_bboxes,
                            im_size_list, cfg):

        cfg_list = [cfg] * len(ext_assigned_gt_inds)

        roi_pair_idx = map(self._get_roi_pairs_single,
                                        ext_assigned_gt_inds,
                                        gt_bboxes,
                                        im_size_list, cfg_list)

        roi_pair_idx = list(roi_pair_idx)
        shift = 0
        for roi_pair, prop in zip(roi_pair_idx, ext_assigned_gt_inds):
            roi_pair[:, 0] += shift
            roi_pair[:, 2] += shift
            shift += len(prop)
        roi_pair_idx = torch.cat(roi_pair_idx)
        return roi_pair_idx

    def get_cluster_masks(self, cluster_pred, det_bboxes, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get cluster segmentation masks from cluster_pred and bboxes.

        Args:
            cluster_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, cluster_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(cluster_pred, torch.Tensor):
            cluster_pred = cluster_pred.cpu().numpy()
        assert isinstance(cluster_pred, np.ndarray)
        # when enabling mixed precision training, cluster_pred may be float16
        # numpy array
        cluster_pred = cluster_pred.astype(np.float32)
        bboxes = det_bboxes.cpu().numpy()[:, :4]

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        cluster_segms = np.zeros((img_h, img_w, self.num_clusters), dtype=np.float32)
        counter = np.zeros((img_h, img_w), dtype=np.uint8)

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            cluster_pred_ = cluster_pred[i].transpose(1, 2, 0)#.argmax(axis=0)
            bbox_mask = imresize(cluster_pred_, (w, h), interpolation='nearest')
            cluster_segms[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = cluster_segms[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] + bbox_mask
            counter[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = counter[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] + 1

        counter[counter<1] = 1
        cluster_segms = cluster_segms / counter[..., None] 
        return cluster_segms

    def get_cluster_mask_ids(self, cluster_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale, roi_id=True):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
   
        cluster_map_6 = self.get_cluster_masks(cluster_pred,  det_bboxes,
                                                rcnn_test_cfg,
                                                ori_shape, scale_factor,
                                                rescale)

        bboxes = det_bboxes.cpu().numpy()

        if isinstance(cluster_pred, torch.Tensor):
            cluster_pred = cluster_pred.cpu().numpy()
        assert isinstance(cluster_pred, np.ndarray)
        # when enabling mixed precision training, cluster_pred may be float16
        # numpy array
        cluster_pred = cluster_pred.astype(np.float32)

        rois = []
        cluster_ids = []
        obj_segms = []
          
        scores = bboxes[:, 4]
        bboxes = bboxes[:, :4]
        labels = det_labels.cpu().numpy() + 1
        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        cluster_map = cluster_map_6[:, :, 1:]
        cluster_map_sum = np.sum(cluster_map, axis=(0, 1))
        
        box_sizes = [max(bbox[2] - bbox[0], bbox[3] - bbox[1]) for bbox in bboxes]
        
        box_size_index = np.argsort(np.array(box_sizes))
        
        for i in box_size_index:
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            score = scores[i]
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            roi_cluster_map_sum = np.sum(cluster_map[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w], axis=(0, 1))
            local_cluster_map_sum = np.sum(cluster_map[max(bbox[1]-100, 0):bbox[1] + h + 100, max(bbox[0]-100, 0):bbox[0] + w + 100], axis=(0, 1))
            
            valid_clusters = roi_cluster_map_sum / (np.ones(self.num_clusters-1) * (w*h))
            valid_cluster_idx = np.where(valid_clusters>0.1)[0]

            fitting_clusters = roi_cluster_map_sum[valid_cluster_idx] / local_cluster_map_sum[valid_cluster_idx]

            fitting_cluster_idx = np.where(fitting_clusters>0.90)[0]

            fitting_cluster_idx = valid_cluster_idx[fitting_cluster_idx]

            roi_cluster_pred = cluster_pred[i][1:, :, :]
            size_list = [(w, h) for _ in roi_cluster_pred]
            roi_cluster_pred = map(self._resize_inst_mask, roi_cluster_pred, size_list)
            roi_cluster_pred = np.stack(list(roi_cluster_pred))

            for c in fitting_cluster_idx:
                im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = (roi_cluster_pred[c, :, :] > 0.5)
                rle = mask_util.encode(np.array(im_mask[:, :, np.newaxis], order='F'))[0]  
                obj_segms.append(rle)
                cluster_ids.append(c)
                rois.append(i)
       
        return obj_segms, cluster_ids, rois, cluster_map_6



