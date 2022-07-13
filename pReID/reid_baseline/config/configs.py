from .default import DefaultConfig


# class Config(DefaultConfig):
#     """
#     mAP 85.8, Rank1 94.1, @epoch 175
#     """
#     def __init__(self):
#         super(Config, self).__init__()
#         self.CFG_NAME = 'baseline'
#         self.DATA_DIR = '/nfs/public/datasets/person_reid/Market-1501-v15.09.15'
#         self.PRETRAIN_CHOICE = 'imagenet'
#         self.PRETRAIN_PATH = '/nfs/public/pretrained_models/resnet50-19c8e357.pth'
#
#         self.LOSS_TYPE = 'triplet+softmax+center'
#         self.TEST_WEIGHT = './output/resnet50_175.pth'
#         self.FLIP_FEATS = 'on'


class Config(DefaultConfig):
    """
    mAP 86.2, Rank1 94.4, @epoch 185
    """

    def __init__(self):
        super(Config, self).__init__()
        self.CFG_NAME        = 'baseline'
        self.DATA_DIR        = '/home/luigy/luigy/datasets/REID/Market-1501-v15.09.15'
        # self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_CHOICE = 'resnet50_ibn_a'
        self.PRETRAIN_PATH   = '/home/luigy/luigy/develop/person-reid-tiny-baseline/weights/resnet50/resnet50_person_reid_128x64.pth'

        self.LOSS_TYPE       = 'triplet+softmax+center'
        self.TEST_WEIGHT     = '/home/luigy/luigy/develop/re3/tracking/pReID/reid_baseline/weights/resnet50_200.pth'
        self.FLIP_FEATS      = 'off'
        self.HARD_FACTOR     = 0.2
        self.RERANKING       = True


# class Config(DefaultConfig):
#     def __init__(self):
#         super(Config, self).__init__()
#         self.CFG_NAME = 'baseline'
#         self.DATA_DIR = '/nfs/public/datasets/person_reid/Market-1501-v15.09.15'
#         self.PRETRAIN_CHOICE = 'imagenet'
#         self.PRETRAIN_PATH = '/nfs/public/pretrained_models/resnet50-19c8e357.pth'
#         self.COS_LAYER = True
#         self.LOSS_TYPE = 'softmax'
#         self.TEST_WEIGHT = './output/resnet50_185.pth'
#         self.FLIP_FEATS = 'off'
#         self.RERANKING = True
