import warnings
import torch


class DefaultConfig(object):
    load_model_path = None  # load model path

    pretrain_model_path = None  # './data/imagenet-vgg-f.mat'

    flag = 'ucm'

    # visualization
    vis_env = None  # visdom env
    vis_port = 8097  # visdom port

    img_dim = 4096
    batch_size = 128
    hidden_dim = 2048
    emb_dim = hidden_dim // 4

    valid = True  # whether to use validation
    valid_freq = 1
    max_epoch = 100

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

    def data(self, flag):
        if flag == 'mir':
            self.dataset = 'flickr25k'
            self.data_path = './data/FLICKR-25K.mat'
            self.db_size = 18015
            self.num_label = 24
            self.query_size = 2000
            self.tag_dim = 1386
            self.training_size = 10000
        if flag == 'nus':
            self.dataset = 'nus-wide'
            self.data_path = './data/NUS-WIDE-TC21.mat'
            self.db_size = 193734
            self.num_label = 21
            self.query_size = 2100
            self.tag_dim = 1000
            self.training_size = 10000
        if flag == 'ucm':
            self.dataset = 'ucm'
            self.data_path = './data/UCM_resnet18_bert_sum_12.h5'
            self.db_size = 9450
            self.num_label = 17
            self.query_size = 1050
            self.tag_dim = 768
            self.training_size = 5250
        if flag == 'rsicd':
            self.dataset = 'rsicd'
            self.data_path = './data/RSICD_resnet18_bert_sum_12.h5'
            self.db_size = 52000
            self.num_label = 31
            self.query_size = 2605
            self.tag_dim = 768
            self.img_dim = 4096
            self.training_size = 30000

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if k == 'flag':
                self.data(v)
            if k == 'proc':
                self.proc = v
            if k == 'device':
                self.device = v
            if k == 'bit':
                self.bit = int(v)
            if not hasattr(self, k):
                warnings.warn("Warning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('Configuration:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and str(k) != 'parse' and str(k) != 'data':
                print('\t{0}: {1}'.format(k, getattr(self, k)))


opt = DefaultConfig()
