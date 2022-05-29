import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.num_classes = 5
Cfg.pin_memory=True
Cfg.num_worker = 4
Cfg.train_batch = 4
Cfg.val_batch_size = 1
Cfg.lr = 0.001
Cfg.epochs = 20
Cfg.scale = 0.5
Cfg.bilinear = False
Cfg.val_interval = 100
Cfg.gpu_id = '3'
Cfg.scheduler_type='cosine_restart'
Cfg.weight_decay = 1e-4
Cfg.momentum = 0.9
#Cfg.TRAIN_OPTIMIZER = 'sgd'
Cfg.TRAIN_OPTIMIZER = 'adam'
Cfg.classes=3
## dataset
Cfg.save_checkpoint = True
Cfg.TRAIN_TENSORBOARD_DIR = './logs'
Cfg.ckpt_dir = os.path.join(_BASE_DIR, 'checkpoints')
