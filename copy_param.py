import os
import shutil
import tempfile
import paddle.fluid as fluid
from mobilenet_v3 import MobileNetV3_Small_
import numpy as np
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module, Parameter,LSTM, Linear
import torch
from mobilenet_v3 import BidirectionalLSTM

def _load_state(path):
    """
    记载paddlepaddle的参数
    :param path:
    :return:
    """
    if os.path.exists(path + '.pdopt'):
        # XXX another hack to ignore the optimizer state
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, os.path.basename(os.path.normpath(path)))
        shutil.copy(path + '.pdparams', dst + '.pdparams')
        state = fluid.io.load_program_state(dst)
        shutil.rmtree(tmp)
    else:
        state = fluid.io.load_program_state(path)
    return state

class InitModel:
    """
    进行参数拷贝
    """
    def __init__(self, paddlepaddle_state, pt_model):
        self.net_pytorch = pt_model
        self.state_pp = paddlepaddle_state
        self.list_layers = list(self.state_pp.keys())
        # self.pass_layer = []
        self.init_model()


    def init_model(self):
        for n, m in self.net_pytorch.named_modules():
            if isinstance(m, BatchNorm2d):
                self.bn_init(n, m)
            elif isinstance(m, Conv2d):
                self.conv_init(n, m)
            # elif isinstance(m, Linear):
            #     self.fc_init(n, m)
            # elif isinstance(m, PReLU):
                # self.prelu_init(n, m)
            elif isinstance(m, BatchNorm1d):
                self.bn_init(n, m)
            # elif isinstance(m, BidirectionalLSTM):
            #     for n1, m1 in m.named_modules():
            #        if isinstance(m1, LSTM):
            #            pass
            #        elif isinstance(m1, Linear):
            #            self.fc_init(n1, m1)

        torch.save(self.net_pytorch.state_dict(),"/home/fuxueping/sdb/CODE/ocr/PaddleOCR/ch_lite/mobilenetV3_small.pth")

    def LSTM(self, layer, m):
        m[0].weight_ih_l0 = 0
        m[0].weight_hh_l0 = 0
        m[0].bias_ih_l0 = 0
        m[0].bias_hh_l0 = 0

        m[1].weight_ih_l0_reverse = 0
        m[1].weight_hh_l0_reverse = 0
        m[1].bias_ih_l0_reverse = 0
        m[1].bias_hh_l0_reverse = 0

    def fc_init(self, layer, m):
        m.weight.data.copy_(torch.FloatTensor(self.state_pp['层名']))
        m.bias.data.copy_(torch.FloatTensor(self.state_pp['层名']))

    def bn_init(self, layer , m):
        for key in self.list_layers:
            if (layer in key) and ('bn' in key):
                print(key) #, ' -- shape: ', self.state_pp[key].shape)
                if 'scale' in key:
                    m.weight.data.copy_(torch.FloatTensor(self.state_pp[key]))
                    self.list_layers.remove(key)
                elif 'offset' in key:
                    m.bias.data.copy_(torch.FloatTensor(self.state_pp[key]))
                    self.list_layers.remove(key)
                elif 'mean' in key:
                    m.running_mean.copy_(torch.FloatTensor(self.state_pp[key]))
                    self.list_layers.remove(key)
                elif 'variance' in key:
                    m.running_var.copy_(torch.FloatTensor(self.state_pp[key]))
                    self.list_layers.remove(key)


    def conv_init(self, layer, m):
        # for pr in net.params:
        layer_ = layer+'_'
        for key in self.list_layers:
            if (layer_ in key) and ('bn' not in key):
                print(key) #, ' -- shape: ', self.state_pp[key].shape)
                if 'weights' in key:
                    m.weight.data.copy_(torch.FloatTensor(self.state_pp[key]))
                elif 'offset' in key:
                    m.bias.data.copy_(self.state_pp[key])
                self.list_layers.remove(key)


    # def prelu_init(self, layer, m):
    #     if layer in self.list_layers:
    #         self.net_caffe.params[layer][0].data[:] = m.weight.data.cpu().numpy()
    #
    # def AdaptiveAvgPool2d_init(self, layer, m):
    #     if layer in self.list_layers:
    #        self.net_caffe.params[layer][0].data[:] = m.weight.data.cpu().numpy()

if __name__=="__main__":
    #加载paddlepaddle的模型
    path = './rec_mv3_crnn/best_accuracy'
    paddlepaddle_state = _load_state(path)

    #加载pytorch的模型
    char_set = open('./char_std_6015.txt', 'r', encoding='utf-8').readlines()
    # char_set = ''.join([ch.strip('\n') for ch in char_set[1:]] + ['卍'])
    n_class = len(char_set)

    pt_model = MobileNetV3_Small_(32, 1, n_class, 192)
    object = InitModel(paddlepaddle_state, pt_model)


