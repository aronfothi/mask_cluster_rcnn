import numpy as np
import os.path as osp
import random
import mmcv
from .custom import CustomDataset
from .youtube import YTVOSDataset
from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         Numpy2Tensor)
from pycocotools.ytvos import YTVOS
from mmcv.parallel import DataContainer as DC

from .registry import DATASETS

@DATASETS.register_module
class HandsDataset(YTVOSDataset):
    CLASSES=('hand')

    