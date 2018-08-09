
import warnings
import mxnet as mx

class DefaultConfig(object):
    env = 'default'
    model = 'VGG16'

    ctx = mx.gpu()

    train_dir = './data/train'
    valid_dir = './data/valid'

    log_file_path = './log/test.log'
    load_file_path = ''

    lr = 0.1
    wd = 5e-4
    momentum = 0.9
    lr_decay = 0.5
    lr_decay_epoch = '14, 20'

    start_epoch = 0
    max_epoch = 20

    log_interval = 100

    batch_size = 64
    num_workers = 0

    val_interval = 1

    save_path = './checkpoints/'

    preload = False

def parse(self, kwargs):
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn('warning: opt has not attribut %s'%k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))

DefaultConfig.parse = parse
opt = DefaultConfig()