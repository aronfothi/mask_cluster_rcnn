import numpy as np
import os.path as osp
import random
import mmcv

import tempfile
import logging
from .custom import CustomDataset
from mmdet.core import eval_map, eval_recalls
from pycocotools.cocoeval import COCOeval
from pycocotools.ytvoseval import YTVOSeval

from pycocotools.ytvos import YTVOS

from mmdet.utils import print_log
from .pipelines import Compose
from .registry import DATASETS


@DATASETS.register_module
class YTVOSDataset(CustomDataset):
    CLASSES=('person','giant_panda','lizard','parrot','skateboard','sedan',
        'ape','dog','snake','monkey','hand','rabbit','duck','cat','cow','fish',
        'train','horse','turtle','bear','motorbike','giraffe','leopard',
        'fox','deer','owl','surfboard','airplane','truck','zebra','tiger',
        'elephant','snowboard','boat','shark','mouse','frog','eagle','earless_seal',
        'tennis_racket')

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 of_prefix=None,
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 seq_len=0,
                 step=1):
        
        # prefix of images path
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.of_prefix = of_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt

        self.seq_len = seq_len

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.of_prefix is None or osp.isabs(self.of_prefix)):
                self.of_prefix = osp.join(self.data_root, self.of_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        
        # load annotations (and proposals)
        self.vid_infos = self.load_annotations(ann_file)

        self.sample_ids = []
        self._frame_ids = {}
        for vid_id, vid_info in self.vid_infos.items():
        
            #video_img_ids = []   
            video_sample_ids = []         
            for frame_id in range(len(vid_info['filenames'])):
                idx = (vid_id, frame_id)
                if test_mode or len(self.get_ann_info(idx)['masks']):
                    video_sample_ids.append(idx)
            if len(video_sample_ids) >= seq_len:
                self._frame_ids[vid_id] = video_sample_ids
                if seq_len>0:
                    self.sample_ids = self.sample_ids + video_sample_ids[0::step]
                    #if test_mode:
                    #    self.sample_ids = self.sample_ids+ [video_sample_ids[-1]]

                else:
                    self.sample_ids = self.sample_ids+ [video_sample_ids[-1]]            
            
        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.sample_ids)

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(self.sample_ids[idx])
        while True:
            data = self.prepare_train_img(self.sample_ids[idx])
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data    

    def load_annotations(self, ann_file):
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.vid_ids = self.ytvos.getVidIds()
        vid_infos = {}
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            vid_infos[i] = info
        return vid_infos

    def get_ann_info(self, idx):
        vid_id, frame_id = idx
        #vid_id = self.vid_infos[idx]['id']
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        ann_info = self.ytvos.loadAnns(ann_ids)
        return self._parse_ann_info(self.get_image_info(idx), ann_info, frame_id)

    def get_image_info(self, idx):
        vid_id, frame_id = idx
        vid_info = self.vid_infos[vid_id]
        return dict(filename=vid_info['filenames'][frame_id],
                    height=vid_info['height'],
                    width=vid_info['width'],
                    video_id=vid_id,
                    frame_id=frame_id)

    def pre_pipeline(self, results, idx, prev_results=None):
        results['img_prefix'] = self.img_prefix
        results['of_prefix'] = self.of_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        video_id, frame_id = idx
        results['video_id'] = video_id
        results['frame_id'] = frame_id
        if prev_results is not None:
            results.update(prev_results)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self.sample_ids), dtype=np.uint8)
        for vid_id, _ in self.sample_ids:
            vid_info = self.vid_infos[vid_id]
            if vid_info['width'] / vid_info['height'] > 1:
                self.flag[vid_id] = 1


    def sample_ref_seq(self, idx):
        vid, fid = idx
        ref_frame_ids = self._frame_ids[vid]
        frame_index = ref_frame_ids.index(idx)
        ref_frame_ids_1 = ref_frame_ids[max(frame_index-(self.seq_len), 0):frame_index]
        ref_frame_ids_2 = ref_frame_ids[frame_index+1:frame_index+(self.seq_len)]
        ref_frame_ids = ref_frame_ids_1 + ref_frame_ids_2
        if len(ref_frame_ids) < self.seq_len-1:
            print('frame_index', frame_index, frame_index-self.seq_len, frame_index+self.seq_len, ref_frame_ids)
            return None, 0
        ref_frame_ids = random.sample(ref_frame_ids, self.seq_len-1)
        ref_frame_ids.append((vid, fid))
        ref_frame_ids.sort()
        frame_index = ref_frame_ids.index(idx)
        return ref_frame_ids, frame_index

    def test_ref_seq(self, idx):
        vid, fid = idx
        ref_frame_ids = self._frame_ids[vid]
        frame_index = ref_frame_ids.index(idx)
        ref_frame_ids = ref_frame_ids[frame_index:frame_index+self.seq_len]
        if len(ref_frame_ids) < self.seq_len:
            print('frame_index', frame_index, frame_index-self.seq_len, frame_index+self.seq_len, ref_frame_ids)
            ref_frame_ids = None
        else:
            ref_frame_ids.sort()
        return ref_frame_ids

    def prepare_train_img(self, idx):
        # prepare a pair of image in a sequence

        seq = []
        flip_keys = ['flip','flip_direction']
        prev_results = None
        samples, ref_frame_index = self.sample_ref_seq(idx)
        if samples is None:            
            return None
        
        for sample_idx in samples:

            img_info = self.get_image_info(sample_idx)
            ann_info = self.get_ann_info(sample_idx)
            results = dict(img_info=img_info, ann_info=ann_info)
            if self.proposals is not None:
                results['proposals'] = self.proposals[idx]
            self.pre_pipeline(results, sample_idx, prev_results=prev_results)
            seq.append(self.pipeline(results))
            if prev_results is None:
                prev_results = {flip_key: results[flip_key] for flip_key in flip_keys}            
        return dict(img=seq[0]['img'], img_meta=seq[0]['img_meta'], inp_seq=seq, ref_frame_index=ref_frame_index)

    def prepare_test_img(self, idx):

        vid, fid = idx
        ref_frame_ids = self._frame_ids[vid] 
        ref_frame_ids.sort()
        #frame_idx = ref_frame_ids.index((str(vid), fid))
        frame_idx = ref_frame_ids.index((vid, fid))

        ref_frame_ids = ref_frame_ids[frame_idx:frame_idx+self.seq_len]
        seq = []
        for sample_idx in ref_frame_ids:

            img_info = self.get_image_info(sample_idx)
            results = dict(img_info=img_info)
            if self.proposals is not None:
                results['proposals'] = self.proposals[idx]
            self.pre_pipeline(results, sample_idx)
            seq.append(self.pipeline(results))

        return dict(img=seq[0]['img'], img_meta=seq[0]['img_meta'], inp_seq=seq)
        
    def _parse_ann_info(self, img_info, ann_info, frame_id):
        """Parse bbox and mask annotation.
        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.
        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            # each ann is a list of masks
            # ann:
            # bbox: list of bboxes
            # segmentation: list of segmentation
            # category_id
            # area: list of area
            bbox = ann['bboxes'][frame_id]
            area = ann['areas'][frame_id]
            segm = ann['segmentations'][frame_id]
            if bbox is None: continue
            x1, y1, w, h = bbox
            if area <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_ids.append(ann['id'])
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentations'][frame_id])
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_ids = np.array(gt_ids, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_ids = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            inst_ids=gt_ids,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)
        
        return ann

    def _segm2json(self, results_list):
        """Dump the detection results to a json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and
                values are corresponding filenames.
        """

        results = []
        for rl_det, rl_seg in results_list:
            
            for d, s in zip(rl_det, rl_seg):
                results.append((d, s))


        json_results = []
        vid_objs = {}
        res_idx = 0
        for idx in range(len(self)):
            # assume results is ordered
            
            vid_id, fr_id = self.sample_ids[idx]
            ref_frame_ids = self._frame_ids[vid_id]
            vid_len = len(ref_frame_ids)
            for frame_id in range(fr_id, min(fr_id + self.seq_len, vid_len)):

                is_last = frame_id == vid_len-1
                
                det, seg = results[res_idx]
                res_idx += 1
                for obj_id in det:
                    bbox = det[obj_id]['bbox']
                    
                    if obj_id in seg:
                        segm = seg[obj_id]
                        label = det[obj_id]['label']
                        
                        if obj_id not in vid_objs:                        
                            vid_objs[obj_id] = {'scores':[],'cats':[], 'segms':{}}
                        vid_objs[obj_id]['scores'].append(bbox[4])
                        vid_objs[obj_id]['cats'].append(label)
                        segm['counts'] = segm['counts'].decode()
                        vid_objs[obj_id]['segms'][frame_id] = segm
                if is_last:
                    # store results of  the current video
                    for obj_id, obj in vid_objs.items():
                        data = dict()

                        data['video_id'] = vid_id
                        data['score'] = np.array(obj['scores']).mean().item()
                        # majority voting for sequence category
                        data['category_id'] = np.bincount(np.array(obj['cats'])).argmax().item() + 1
                        vid_seg = []
                        for fid in range(frame_id + 1):
                            if fid in obj['segms']:
                                vid_seg.append(obj['segms'][fid])
                            else:
                                vid_seg.append(None)
                        data['segmentations'] = vid_seg
                        first = False
                        json_results.append(data)
                    vid_objs = {}

        return [], json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = '{}.{}.json'.format(outfile_prefix, 'bbox')
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'bbox')
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = '{}.{}.json'.format(outfile_prefix, 'bbox')
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'bbox')
            result_files['segm'] = '{}.{}.json'.format(outfile_prefix, 'segm')
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'proposal')
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        '''
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))
        '''

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir
    
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        """Evaluation in COCO protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str: float]
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError('metric {} is not supported'.format(metric))

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = {}
        ytvosGt = self.ytvos
        for metric in metrics:
            msg = 'Evaluating {}...'.format(metric)
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
                    log_msg.append('\nAR@{}\t{:.4f}'.format(num, ar[i]))
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError('{} is not in results'.format(metric))
            try:
                ytvosDt = ytvosGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            ytvosEval = YTVOSeval(ytvosGt, ytvosDt, iou_type)
            vid_ids = self.ytvos.getVidIds()
            ytvosEval.params.vidIds = vid_ids
            if metric == 'proposal':
                ytvosEval.params.useCats = 0
                ytvosEval.params.maxDets = list(proposal_nums)
                ytvosEval.evaluate()
                ytvosEval.accumulate()
                ytvosEval.summarize()
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
                    'AR_l@1000'
                ]
                for i, item in enumerate(metric_items):
                    val = float('{:.3f}'.format(ytvosEval.stats[i + 6]))
                    eval_results[item] = val
            else:
                ytvosEval.evaluate()
                ytvosEval.accumulate()
                ytvosEval.summarize()
                if classwise:  # Compute per-category AP
                    pass  # TODO
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]
                for i in range(len(metric_items)):
                    key = '{}_{}'.format(metric, metric_items[i])
                    val = float('{:.3f}'.format(ytvosEval.stats[i]))
                    eval_results[key] = val
                eval_results['{}_mAP_copypaste'.format(metric)] = (
                    '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    '{ap[4]:.3f} {ap[5]:.3f}').format(ap=ytvosEval.stats[:6])
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results