import torch
from functools import partial
from easydict import EasyDict as edict
from albumentations import *

from isegm.data.datasets import *
from isegm.model.losses import *
from isegm.data.transforms import *
from isegm.engine.trainer import ISTrainer
from isegm.model.metrics import AdaptiveIoU
from isegm.data.points_sampler import MultiPointSampler
from isegm.utils.log import logger
from isegm.model import initializer

from isegm.model.is_hrnet_model import HRNetModel
from isegm.model.is_deeplab_model import DeeplabModel
# from isegm.model.is_segformer_model import SegformerModel
from isegm.model.is_hrformer_model import HRFormerModel
from isegm.model.is_swinformer_model import SwinformerModel
from isegm.model.is_plainvit_model import PlainVitModel

from isegm.model.is_vitdetr_gaussianvector_model import VitGaussianVector_Model
from isegm.model.is_vitdetr_gaussianvector_edloss_model import VitGaussianVector_ed_Model
from isegm.model.is_vitdetr_multigaussianvector_edloss_model import VitMultiGaussianVector_ed_Model
from isegm.model.is_vitdetr_multigaussianvector_only_edloss_model import VitMultiGaussianVector_only_ed_Model
from isegm.model.is_vitdetr_learnablevector_model import VitLearnableVector_Model
from isegm.model.is_vitdetr_mmvector_model import VitMMVector_Model

from isegm.model.is_deeplab_gaussianvector_model import Deeplab_GV_Model
from isegm.model.is_hrnet_gaussianvector_model import HRNet_GV_Model
from isegm.model.is_segformer_gaussianvector_model import SegFormerModel
