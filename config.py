import warnings
import torch


class DefaultConfig(object):
    load_model_path = None  # load model path

    pretrain_model_path = None  # './data/imagenet-vgg-f.mat'

    # visualization
    vis_env = None  # visdom env
    vis_port = 8097  # visdom port

    # for ucm dataset
    flag = 'ucm'
    dataset = 'ucm'
    data_path = './data/resnet18_UCM_multi_label.h5'
    db_size = 9450
    num_label = 17
    tag_dim = 266
    img_dim = 4096
    query_size = 1050

    # for flickr25k dataset
    # dataset = 'flickr25k'
    # data_path = './data/FLICKR-25K.mat'
    # db_size = 18015
    # num_label = 24
    # tag_dim = 1386
    # query_size = 2000

    # for nus-wide dataset
    # dataset = 'nus-wide'
    # data_path = './data/dcmh-nus-wide'
    # db_size = 193734
    # num_label = 21
    # tag_dim = 1000
    # query_size = 2100

    training_size = 7350
    batch_size = 128
    hidden_dim = 2048
    emb_dim = hidden_dim // 4

    valid = True  # whether to use validation
    valid_freq = 1
    max_epoch = 5

    bit = 64  # final binary code length
    lr = 0.0001  # initial learning rate

    device = 'cuda:0'  # 'cuda:0'  # if `device` is not None then use cpu for default

    # hyper-parameters
    alpha = 1
    beta = 0.1
    gamma = 0.00001
    eta = 0.01
    mu = 100
    delta = 0.5

    lambd = 0.8
    margin = 0.3

    # for program debug
    debug = False
    data_enhance = False

    proc = None

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if k == 'proc':
                self.proc = v
            if k == 'device':
                self.device = v
            if k == 'flag':
                self.flag = v
            if k == 'bit':
                self.bit = int(v)
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('Configuration:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and str(k) != 'parse':
                print('\t{0}: {1}'.format(k, getattr(self, k)))


opt = DefaultConfig()
