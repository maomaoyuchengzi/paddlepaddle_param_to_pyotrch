import torch.nn as nn
import torch.nn.functional as F
# import copy

#----------by fxp
#----------2020_08_10
class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class MobileNetV3_Small_(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False,inplanes = 16, scale = 0.5):
        super(MobileNetV3_Small_, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, groups=1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv1_bn_activation = hswish()

        # 0 [3, 16, 16, True, 'relu', (2, 1)],
        self.conv2_expand = nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv2_expand_bn = nn.BatchNorm2d(8)
        self.conv2_expand_bn_activation = nn.ReLU(inplace=True)

        self.conv2_depthwise = nn.Conv2d(8, 8, kernel_size=3, stride=(2, 1), padding=1, groups=8, bias=False)
        self.conv2_depthwise_bn = nn.BatchNorm2d(8)
        self.conv2_depthwise_bn_activation = nn.ReLU(inplace=True)

        # se
        self.conv2_se_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv2_se1 = nn.Conv2d(8, 2, kernel_size=1)
        self.conv2_se1_activation = nn.ReLU(inplace=True)
        self.conv2_se2 = nn.Conv2d(2, 8, kernel_size=1)
        self.conv2_se2_activation = hsigmoid()
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)

        self.conv2_linear = nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv2_linear_bn = nn.BatchNorm2d(8)

        # 1 [3, 72, 24, False, 'relu', (2, 1)],
        self.conv3_expand = nn.Conv2d(8, 40, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv3_expand_bn = nn.BatchNorm2d(40)
        self.conv3_expand_bn_activation = nn.ReLU(inplace=True)

        self.conv3_depthwise = nn.Conv2d(40, 40, kernel_size=3, stride=(2, 1), padding=1, groups=40, bias=False)
        self.conv3_depthwise_bn = nn.BatchNorm2d(40)
        self.conv3_depthwise_bn_activation = nn.ReLU(inplace=True)

        self.conv3_linear = nn.Conv2d(40, 16, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv3_linear_bn = nn.BatchNorm2d(16)

        # for 2 [3, 88, 24, False, 'relu', 1],
        self.conv4_expand = nn.Conv2d(16, 48, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv4_expand_bn = nn.BatchNorm2d(48)
        self.conv4_expand_bn_activation = nn.ReLU(inplace=True)

        self.conv4_depthwise = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, groups=48, bias=False)
        self.conv4_depthwise_bn = nn.BatchNorm2d(48)
        self.conv4_depthwise_bn_activation = nn.ReLU(inplace=True)

        self.conv4_linear = nn.Conv2d(48, 16, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv4_linear_bn = nn.BatchNorm2d(16)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)

        # for 3 [5, 96, 40, True, 'hard_swish', (2, 1)],
        # use_se
        self.conv5_expand = nn.Conv2d(16, 48, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv5_expand_bn = nn.BatchNorm2d(48)
        self.conv5_expand_bn_activation = hswish()

        self.conv5_depthwise = nn.Conv2d(48, 48, kernel_size=5, stride=(2, 1), padding=2, groups=48, bias=False)
        self.conv5_depthwise_bn = nn.BatchNorm2d(48)
        self.conv5_depthwise_bn_activation = hswish()

        # se
        self.conv5_se_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv5_se1 = nn.Conv2d(48, 12, kernel_size=1)
        self.conv5_se1_activation = nn.ReLU(inplace=True)
        self.conv5_se2 = nn.Conv2d(12, 48, kernel_size=1)
        self.conv5_se2_activation = hsigmoid()
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)

        self.conv5_linear = nn.Conv2d(48, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv5_linear_bn = nn.BatchNorm2d(24)

        # for 4 [5, 240, 40, True, 'hard_swish', 1],
        # use_se
        self.conv6_expand = nn.Conv2d(24, 120, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv6_expand_bn = nn.BatchNorm2d(120)
        self.conv6_expand_bn_activation = hswish()

        self.conv6_depthwise = nn.Conv2d(120, 120, kernel_size=5, stride=1, padding=2, groups=120, bias=False)
        self.conv6_depthwise_bn = nn.BatchNorm2d(120)
        self.conv6_depthwise_bn_activation = hswish()

        # se
        self.conv6_se_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv6_se1 = nn.Conv2d(120, 30, kernel_size=1)
        self.conv6_se1_activation = nn.ReLU(inplace=True)
        self.conv6_se2 = nn.Conv2d(30, 120, kernel_size=1)
        self.conv6_se2_activation = hsigmoid()
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)

        self.conv6_linear = nn.Conv2d(120, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv6_linear_bn = nn.BatchNorm2d(24)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)

        # for 5 [5, 240, 40, True, 'hard_swish', 1],
        # use_se
        self.conv7_expand = nn.Conv2d(24, 120, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv7_expand_bn = nn.BatchNorm2d(120)
        self.conv7_expand_bn_activation = hswish()

        self.conv7_depthwise = nn.Conv2d(120, 120, kernel_size=5, stride=1, padding=2, groups=120, bias=False)
        self.conv7_depthwise_bn = nn.BatchNorm2d(120)
        self.conv7_depthwise_bn_activation = hswish()

        # se
        self.conv7_se_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv7_se1 = nn.Conv2d(120, 30, kernel_size=1)
        self.conv7_se1_activation = nn.ReLU(inplace=True)
        self.conv7_se2 = nn.Conv2d(30, 120, kernel_size=1)
        self.conv7_se2_activation = hsigmoid()
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)

        self.conv7_linear = nn.Conv2d(120, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv7_linear_bn = nn.BatchNorm2d(24)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)

        # for 6 [5, 120, 48, True, 'hard_swish', 1],
        self.conv8_expand = nn.Conv2d(24, 64, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv8_expand_bn = nn.BatchNorm2d(64)
        self.conv8_expand_bn_activation = hswish()

        self.conv8_depthwise = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, groups=64, bias=False)
        self.conv8_depthwise_bn = nn.BatchNorm2d(64)
        self.conv8_depthwise_bn_activation = hswish()

        # se
        self.conv8_se_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv8_se1 = nn.Conv2d(64, 16, kernel_size=1)
        self.conv8_se1_activation = nn.ReLU(inplace=True)
        self.conv8_se2 = nn.Conv2d(16, 64, kernel_size=1)
        self.conv8_se2_activation = hsigmoid()
        # scale = fluid.layers.elementwise_mul(x=input, y=conv8, axis=0)

        self.conv8_linear = nn.Conv2d(64, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv8_linear_bn = nn.BatchNorm2d(24)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)

        # for 7 [5, 144, 48, True, 'hard_swish', 1],
        self.conv9_expand = nn.Conv2d(24, 72, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv9_expand_bn = nn.BatchNorm2d(72)
        self.conv9_expand_bn_activation = hswish()

        self.conv9_depthwise = nn.Conv2d(72, 72, kernel_size=5, stride=1, padding=2, groups=72, bias=False)
        self.conv9_depthwise_bn = nn.BatchNorm2d(72)
        self.conv9_depthwise_bn_activation = hswish()

        # se
        self.conv9_se_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv9_se1 = nn.Conv2d(72, 18, kernel_size=1)
        self.conv9_se1_activation = nn.ReLU(inplace=True)
        self.conv9_se2 = nn.Conv2d(18, 72, kernel_size=1)
        self.conv9_se2_activation = hsigmoid()
        # scale = fluid.layers.elementwise_mul(x=input, y=conv9, axis=0)

        self.conv9_linear = nn.Conv2d(72, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv9_linear_bn = nn.BatchNorm2d(24)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)

        # for 8 [5, 288, 96, True, 'hard_swish', (2, 1)],
        self.conv10_expand = nn.Conv2d(24, 144, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv10_expand_bn = nn.BatchNorm2d(144)
        self.conv10_expand_bn_activation = hswish()

        self.conv10_depthwise = nn.Conv2d(144, 144, kernel_size=5, stride=(2, 1), padding=2, groups=144, bias=False)
        self.conv10_depthwise_bn = nn.BatchNorm2d(144)
        self.conv10_depthwise_bn_activation = hswish()

        # se
        self.conv10_se_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv10_se1 = nn.Conv2d(144, 36, kernel_size=1)
        self.conv10_se1_activation = nn.ReLU(inplace=True)
        self.conv10_se2 = nn.Conv2d(36, 144, kernel_size=1)
        self.conv10_se2_activation = hsigmoid()
        # scale = fluid.layers.elementwise_mul(x=input, y=conv10, axis=0)

        self.conv10_linear = nn.Conv2d(144, 48, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv10_linear_bn = nn.BatchNorm2d(48)

        # for 9 [5, 576, 96, True, 'hard_swish', 1],
        self.conv11_expand = nn.Conv2d(48, 288, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv11_expand_bn = nn.BatchNorm2d(288)
        self.conv11_expand_bn_activation = hswish()

        self.conv11_depthwise = nn.Conv2d(288, 288, kernel_size=5, stride=1, padding=2, groups=96, bias=False)
        self.conv11_depthwise_bn = nn.BatchNorm2d(288)
        self.conv11_depthwise_bn_activation = hswish()

        # se
        self.conv11_se_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv11_se1 = nn.Conv2d(288, 72, kernel_size=1)
        self.conv11_se1_activation = nn.ReLU(inplace=True)
        self.conv11_se2 = nn.Conv2d(72, 288, kernel_size=1)
        self.conv11_se2_activation = hsigmoid()
        # scale = fluid.layers.elementwise_mul(x=input, y=conv11, axis=0)

        self.conv11_linear = nn.Conv2d(288, 48, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv11_linear_bn = nn.BatchNorm2d(48)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)

        # for 10 [5, 576, 96, True, 'hard_swish', 1],
        # use_se
        self.conv12_expand = nn.Conv2d(48, 288, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv12_expand_bn = nn.BatchNorm2d(288)
        self.conv12_expand_bn_activation = hswish()

        self.conv12_depthwise = nn.Conv2d(288, 288, kernel_size=5, stride=1, padding=2, groups=288, bias=False)
        self.conv12_depthwise_bn = nn.BatchNorm2d(288)
        self.conv12_depthwise_bn_activation = hswish()

        # se
        self.conv12_se_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv12_se1 = nn.Conv2d(288, 72, kernel_size=1)
        self.conv12_se1_activation = nn.ReLU(inplace=True)
        self.conv12_se2 = nn.Conv2d(72, 288, kernel_size=1)
        self.conv12_se2_activation = hsigmoid()
        # scale = fluid.layers.elementwise_mul(x=input, y=conv12, axis=0)

        self.conv12_linear = nn.Conv2d(288, 48, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv12_linear_bn = nn.BatchNorm2d(48)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)

        # last
        self.conv_last_expand = nn.Conv2d(48, 288, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv_last_expand_bn = nn.BatchNorm2d(288)
        self.conv_last_expand_bn_activation = hswish()

        self.max_pool = nn.MaxPool2d(2, stride=2)  # 括号内第一个参数是:窗口的大小,第二个是移动的步长距离

        self.rnn = nn.Sequential(
            BidirectionalLSTM(288, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        self.rnn1 = BidirectionalLSTM(288, nh, nh)
        self.rnn2 = BidirectionalLSTM(nh, nh, nclass)
        pass

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv1_bn(output)
        output = self.conv1_bn_activation(output)

        # 0 [3, 16, 16, True, 'relu', (2, 1)],
        output = self.conv2_expand(output)
        output = self.conv2_expand_bn(output)
        output = self.conv2_expand_bn_activation(output)

        output = self.conv2_depthwise(output)
        output = self.conv2_depthwise_bn(output)
        output = self.conv2_depthwise_bn_activation(output)

        # se
        input = output
        output = self.conv2_se_avg(output)
        output = self.conv2_se1(output)
        output = self.conv2_se1_activation(output)
        output = self.conv2_se2(output)
        output = self.conv2_se2_activation(output)
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)
        output = output*input

        output = self.conv2_linear(output)
        output = self.conv2_linear_bn(output)

        # 1 [3, 72, 24, False, 'relu', (2, 1)],
        output = self.conv3_expand(output)
        output = self.conv3_expand_bn(output)
        output = self.conv3_expand_bn_activation(output)

        output = self.conv3_depthwise(output)
        output = self.conv3_depthwise_bn(output)
        output = self.conv3_depthwise_bn_activation(output)

        output = self.conv3_linear(output)
        output = self.conv3_linear_bn(output)

        # for 2 [3, 88, 24, False, 'relu', 1],
        input1 = output
        output = self.conv4_expand(output)
        output = self.conv4_expand_bn(output)
        output = self.conv4_expand_bn_activation(output)

        output = self.conv4_depthwise(output)
        output = self.conv4_depthwise_bn(output)
        output = self.conv4_depthwise_bn_activation(output)

        output = self.conv4_linear(output)
        output = self.conv4_linear_bn(output)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        output = input1 + output

        # for 3 [5, 96, 40, True, 'hard_swish', (2, 1)],
        # use_se
        output = self.conv5_expand(output)
        output = self.conv5_expand_bn(output)
        output = self.conv5_expand_bn_activation(output)

        output = self.conv5_depthwise(output)
        output = self.conv5_depthwise_bn(output)
        output = self.conv5_depthwise_bn_activation(output)

        # se
        input = output
        output = self.conv5_se_avg(output)
        output = self.conv5_se1(output)
        output = self.conv5_se1_activation(output)
        output = self.conv5_se2(output)
        output = self.conv5_se2_activation(output)
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)
        output = input*output

        output = self.conv5_linear(output)
        output = self.conv5_linear_bn(output)

        # for 4 [5, 240, 40, True, 'hard_swish', 1],
        # use_se
        input1 = output
        output = self.conv6_expand(output)
        output = self.conv6_expand_bn(output)
        output = self.conv6_expand_bn_activation(output)

        output = self.conv6_depthwise(output)
        output = self.conv6_depthwise_bn(output)
        output = self.conv6_depthwise_bn_activation(output)

        # se
        input = output
        output = self.conv6_se_avg(output)
        output = self.conv6_se1(output)
        output = self.conv6_se1_activation(output)
        output = self.conv6_se2(output)
        output = self.conv6_se2_activation(output)
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)
        output = input*output

        output = self.conv6_linear(output)
        output = self.conv6_linear_bn(output)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        output = input1 + output

        # for 5 [5, 240, 40, True, 'hard_swish', 1],
        # use_se
        input1 = output
        output = self.conv7_expand(output)
        output = self.conv7_expand_bn(output)
        output = self.conv7_expand_bn_activation(output)

        output = self.conv7_depthwise(output)
        output = self.conv7_depthwise_bn(output)
        output = self.conv7_depthwise_bn_activation(output)

        # se
        input = output
        output = self.conv7_se_avg(output)
        output = self.conv7_se1(output)
        output = self.conv7_se1_activation(output)
        output = self.conv7_se2(output)
        output = self.conv7_se2_activation(output)
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)
        output = input*output

        output = self.conv7_linear(output)
        output = self.conv7_linear_bn(output)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        output = input1 + output

        # for 6 [5, 120, 48, True, 'hard_swish', 1],
        input1 = output
        output = self.conv8_expand(output)
        output = self.conv8_expand_bn(output)
        output = self.conv8_expand_bn_activation(output)

        output = self.conv8_depthwise(output)
        output = self.conv8_depthwise_bn(output)
        output = self.conv8_depthwise_bn_activation(output)

        # se
        input = output
        output = self.conv8_se_avg(output)
        output = self.conv8_se1(output)
        output = self.conv8_se1_activation(output)
        output = self.conv8_se2(output)
        output = self.conv8_se2_activation(output)
        # scale = fluid.layers.elementwise_mul(x=input, y=conv8, axis=0)
        output = input*output

        output = self.conv8_linear(output)
        output = self.conv8_linear_bn(output)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        output = input1 + output

        # for 7 [5, 144, 48, True, 'hard_swish', 1],
        input1 = output
        output = self.conv9_expand(output)
        output = self.conv9_expand_bn(output)
        output = self.conv9_expand_bn_activation(output)

        output = self.conv9_depthwise(output)
        output = self.conv9_depthwise_bn(output)
        output = self.conv9_depthwise_bn_activation(output)

        # se
        input = output
        output = self.conv9_se_avg(output)
        output = self.conv9_se1(output)
        output = self.conv9_se1_activation(output)
        output = self.conv9_se2(output)
        output = self.conv9_se2_activation(output)
        # scale = fluid.layers.elementwise_mul(x=input, y=conv9, axis=0)
        output = input * output

        output = self.conv9_linear(output)
        output = self.conv9_linear_bn(output)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        output = input1 + output

        # for 8 [5, 288, 96, True, 'hard_swish', (2, 1)],
        output = self.conv10_expand (output)
        output = self.conv10_expand_bn(output)
        output = self.conv10_expand_bn_activation(output)

        output = self.conv10_depthwise(output)
        output = self.conv10_depthwise_bn (output)
        output = self.conv10_depthwise_bn_activation (output)

        # se
        input = output
        output = self.conv10_se_avg (output)
        output = self.conv10_se1 (output)
        output = self.conv10_se1_activation(output)
        output = self.conv10_se2(output)
        output = self.conv10_se2_activation(output)
        # scale = fluid.layers.elementwise_mul(x=input, y=conv10, axis=0)
        output = input * output

        output = self.conv10_linear(output)
        output = self.conv10_linear_bn(output)

        # for 9 [5, 576, 96, True, 'hard_swish', 1],
        input1 = output
        output = self.conv11_expand(output)
        output = self.conv11_expand_bn(output)
        output = self.conv11_expand_bn_activation(output)

        output = self.conv11_depthwise(output)
        output = self.conv11_depthwise_bn(output)
        output = self.conv11_depthwise_bn_activation(output)

        # se
        input= output
        output = self.conv11_se_avg(output)
        output = self.conv11_se1(output)
        output = self.conv11_se1_activation(output)
        output = self.conv11_se2(output)
        output = self.conv11_se2_activation(output)
        # scale = fluid.layers.elementwise_mul(x=input, y=conv11, axis=0)
        output = input * output

        output = self.conv11_linear(output)
        output = self.conv11_linear_bn(output)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        output = input1 + output

        # for 10 [5, 576, 96, True, 'hard_swish', 1],
        # use_se
        input1 = output
        output = self.conv12_expand(output)
        output = self.conv12_expand_bn(output)
        output = self.conv12_expand_bn_activation(output)

        output = self.conv12_depthwise(output)
        output = self.conv12_depthwise_bn(output)
        output = self.conv12_depthwise_bn_activation(output)

        # se
        input = output
        output = self.conv12_se_avg (output)
        output = self.conv12_se1(output)
        output = self.conv12_se1_activation(output)
        output = self.conv12_se2 (output)
        output = self.conv12_se2_activation(output)
        # scale = fluid.layers.elementwise_mul(x=input, y=conv12, axis=0)
        output = input*output

        output = self.conv12_linear(output)
        output = self.conv12_linear_bn(output)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        output = input1 + output

        # last
        output = self.conv_last_expand (output)
        output = self.conv_last_expand_bn(output)
        output = self.conv_last_expand_bn_activation(output)

        output = self.max_pool(output)
        print(output.shape)

        b, c, h, w = output.size()
        assert h == 1, "the height of conv must be 1"
        conv = output.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        # output = self.rnn(conv)  # 294*1*5991
        output = self.rnn1(conv)  # 294*1*5991
        output = self.rnn2(output)  # 294*1*5991
        # output = self.rnn(conv)

        return output



class MobileNetV3_Large_(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False,inplanes = 16, scale = 0.5):
        super(MobileNetV3_Large_, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, groups=1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv1_bn_activation = hswish()

        # 0
        self.conv2_expand = nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv2_expand_bn = nn.BatchNorm2d(8)
        self.conv2_expand_bn_activation = nn.ReLU(inplace=True)

        self.conv2_depthwise = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=8, bias=False)
        self.conv2_depthwise_bn = nn.BatchNorm2d(8)
        self.conv2_depthwise_bn_activation = nn.ReLU(inplace=True)

        self.conv2_linear = nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv2_linear_bn = nn.BatchNorm2d(8)

        # self.conv2_linear_elementwise = fluid.layers.elementwise_add(x=input = A, y = conv2, act = None)  # 逐个元素相加

        # 1
        self.conv3_expand = nn.Conv2d(8, 32, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv3_expand_bn = nn.BatchNorm2d(32)
        self.conv3_expand_bn_activation = nn.ReLU(inplace=True)

        self.conv3_depthwise = nn.Conv2d(32, 32, kernel_size=3, stride=(2, 1), padding=1, groups=32, bias=False)
        self.conv3_depthwise_bn = nn.BatchNorm2d(32)
        self.conv3_depthwise_bn_activation = nn.ReLU(inplace=True)

        self.conv3_linear = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv3_linear_bn = nn.BatchNorm2d(16)

        # for 2
        self.conv4_expand = nn.Conv2d(16, 40, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv4_expand_bn = nn.BatchNorm2d(40)
        self.conv4_expand_bn_activation = nn.ReLU(inplace=True)

        self.conv4_depthwise = nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1, groups=40, bias=False)
        self.conv4_depthwise_bn = nn.BatchNorm2d(40)
        self.conv4_depthwise_bn_activation = nn.ReLU(inplace=True)

        self.conv4_linear = nn.Conv2d(40, 16, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv4_linear_bn = nn.BatchNorm2d(16)

        # self.conv4_linear_elementwise=fluid.layers.elementwise_add(x=input, y=conv4, act=None)

        # for 3 [5,72,40,True,'relu',(2,1)]
        # use_se
        self.conv5_expand = nn.Conv2d(16, 40, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv5_expand_bn = nn.BatchNorm2d(40)
        self.conv5_expand_bn_activation = nn.ReLU(inplace=True)

        self.conv5_depthwise = nn.Conv2d(40, 40, kernel_size=5, stride=(2, 1), padding=2, groups=40, bias=False)
        self.conv5_depthwise_bn = nn.BatchNorm2d(40)
        self.conv5_depthwise_bn_activation = nn.ReLU(inplace=True)

        # se
        self.conv5_se_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv5_se1 = nn.Conv2d(40, 10, kernel_size=1)
        self.conv5_se1_activation = nn.ReLU(inplace=True)
        self.conv5_se2 = nn.Conv2d(10, 40, kernel_size=1)
        self.conv5_se2_activation = hsigmoid()
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)

        self.conv5_linear = nn.Conv2d(40, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv5_linear_bn = nn.BatchNorm2d(24)

        # for 4 [5,120,40,True,'relu',1]
        # use_se
        self.conv6_expand = nn.Conv2d(24, 64, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv6_expand_bn = nn.BatchNorm2d(64)
        self.conv6_expand_bn_activation = nn.ReLU(inplace=True)

        self.conv6_depthwise = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, groups=64, bias=False)
        self.conv6_depthwise_bn = nn.BatchNorm2d(64)
        self.conv6_depthwise_bn_activation = nn.ReLU(inplace=True)

        # se
        self.conv6_se_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv6_se1 = nn.Conv2d(64, 16, kernel_size=1)
        self.conv6_se1_activation = nn.ReLU(inplace=True)
        self.conv6_se2 = nn.Conv2d(16, 64, kernel_size=1)
        self.conv6_se2_activation = hsigmoid()
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)

        self.conv6_linear = nn.Conv2d(64, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv6_linear_bn = nn.BatchNorm2d(24)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)

        # for 5 [5,120,40,True,'relu',1]
        # use_se
        self.conv7_expand = nn.Conv2d(24, 64, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv7_expand_bn = nn.BatchNorm2d(64)
        self.conv7_expand_bn_activation = nn.ReLU(inplace=True)

        self.conv7_depthwise = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, groups=64, bias=False)
        self.conv7_depthwise_bn = nn.BatchNorm2d(64)
        self.conv7_depthwise_bn_activation = nn.ReLU(inplace=True)

        # se
        self.conv7_se_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv7_se1 = nn.Conv2d(64, 16, kernel_size=1)
        self.conv7_se1_activation = nn.ReLU(inplace=True)
        self.conv7_se2 = nn.Conv2d(16, 64, kernel_size=1)
        self.conv7_se2_activation = hsigmoid()
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)

        self.conv7_linear = nn.Conv2d(64, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv7_linear_bn = nn.BatchNorm2d(24)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)

        # for 6 [3,240,80,False,'hard_swish',1]
        self.conv8_expand = nn.Conv2d(24, 120, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv8_expand_bn = nn.BatchNorm2d(120)
        self.conv8_expand_bn_activation = hswish()

        self.conv8_depthwise = nn.Conv2d(120, 120, kernel_size=3, stride=1, padding=1, groups=120, bias=False)
        self.conv8_depthwise_bn = nn.BatchNorm2d(120)
        self.conv8_depthwise_bn_activation = hswish()

        self.conv8_linear = nn.Conv2d(120, 40, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv8_linear_bn = nn.BatchNorm2d(40)

        # for 7 [3,240,80,False,'hard_swish',1]
        self.conv9_expand = nn.Conv2d(40, 104, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv9_expand_bn = nn.BatchNorm2d(104)
        self.conv9_expand_bn_activation = hswish()

        self.conv9_depthwise = nn.Conv2d(104, 104, kernel_size=3, stride=1, padding=1, groups=104, bias=False)
        self.conv9_depthwise_bn = nn.BatchNorm2d(104)
        self.conv9_depthwise_bn_activation = hswish()

        self.conv9_linear = nn.Conv2d(104, 40, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv9_linear_bn = nn.BatchNorm2d(40)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)

        # for 8 [3,184,80,False,'hard_swish',1]
        self.conv10_expand = nn.Conv2d(40, 96, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv10_expand_bn = nn.BatchNorm2d(96)
        self.conv10_expand_bn_activation = hswish()

        self.conv10_depthwise = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, groups=96, bias=False)
        self.conv10_depthwise_bn = nn.BatchNorm2d(96)
        self.conv10_depthwise_bn_activation = hswish()

        self.conv10_linear = nn.Conv2d(96, 40, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv10_linear_bn = nn.BatchNorm2d(40)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)

        # for 9 [3,184,80,False,'hard_swish',1]
        self.conv11_expand = nn.Conv2d(40, 96, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv11_expand_bn = nn.BatchNorm2d(96)
        self.conv11_expand_bn_activation = hswish()

        self.conv11_depthwise = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, groups=96, bias=False)
        self.conv11_depthwise_bn = nn.BatchNorm2d(96)
        self.conv11_depthwise_bn_activation = hswish()

        self.conv11_linear = nn.Conv2d(96, 40, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv11_linear_bn = nn.BatchNorm2d(40)

        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)

        # for 10 [3,480,112,False,'hard_swish',1]
        # use_se
        self.conv12_expand = nn.Conv2d(40, 240, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv12_expand_bn = nn.BatchNorm2d(240)
        self.conv12_expand_bn_activation = hswish()

        self.conv12_depthwise = nn.Conv2d(240, 240, kernel_size=3, stride=1, padding=1, groups=240, bias=False)
        self.conv12_depthwise_bn = nn.BatchNorm2d(240)
        self.conv12_depthwise_bn_activation = hswish()

        # se
        self.conv12_se_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv12_se1 = nn.Conv2d(240, 60, kernel_size=1)
        self.conv12_se1_activation = nn.ReLU(inplace=True)
        self.conv12_se2 = nn.Conv2d(60, 240, kernel_size=1)
        self.conv12_se2_activation = hsigmoid()
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)

        self.conv12_linear = nn.Conv2d(240, 56, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv12_linear_bn = nn.BatchNorm2d(56)

        # for 11 [3,672,112,False,'hard_swish',1]
        # use_se
        self.conv13_expand = nn.Conv2d(56, 336, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv13_expand_bn = nn.BatchNorm2d(336)
        self.conv13_expand_bn_activation = hswish()

        self.conv13_depthwise = nn.Conv2d(336, 336, kernel_size=3, stride=1, padding=1, groups=336, bias=False)
        self.conv13_depthwise_bn = nn.BatchNorm2d(336)
        self.conv13_depthwise_bn_activation = hswish()

        # se
        self.conv13_se_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv13_se1 = nn.Conv2d(336, 84, kernel_size=1)
        self.conv13_se1_activation = nn.ReLU(inplace=True)
        self.conv13_se2 = nn.Conv2d(84, 336, kernel_size=1)
        self.conv13_se2_activation = hsigmoid()
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)

        self.conv13_linear = nn.Conv2d(336, 56, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv13_linear_bn = nn.BatchNorm2d(56)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)

        # for 12 [3,672,160,False,'hard_swish',1]
        # use_se
        self.conv14_expand = nn.Conv2d(56, 336, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv14_expand_bn = nn.BatchNorm2d(336)
        self.conv14_expand_bn_activation = hswish()

        self.conv14_depthwise = nn.Conv2d(336, 336, kernel_size=5, stride=(2, 1), padding=2, groups=336, bias=False)
        self.conv14_depthwise_bn = nn.BatchNorm2d(336)
        self.conv14_depthwise_bn_activation = hswish()

        # se
        self.conv14_se_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv14_se1 = nn.Conv2d(336, 84, kernel_size=1)
        self.conv14_se1_activation = nn.ReLU(inplace=True)
        self.conv14_se2 = nn.Conv2d(84, 336, kernel_size=1)
        self.conv14_se2_activation = hsigmoid()
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)

        self.conv14_linear = nn.Conv2d(336, 80, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv14_linear_bn = nn.BatchNorm2d(80)

        # for 13 [3,960,160,False,'hard_swish',1]
        # use_se
        self.conv15_expand = nn.Conv2d(80, 480, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv15_expand_bn = nn.BatchNorm2d(480)
        self.conv15_expand_bn_activation = hswish()

        self.conv15_depthwise = nn.Conv2d(480, 480, kernel_size=5, stride=1, padding=2, groups=480, bias=False)
        self.conv15_depthwise_bn = nn.BatchNorm2d(480)
        self.conv15_depthwise_bn_activation = hswish()

        # se
        self.conv15_se_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv15_se1 = nn.Conv2d(480, 120, kernel_size=1)
        self.conv15_se1_activation = nn.ReLU(inplace=True)
        self.conv15_se2 = nn.Conv2d(120, 480, kernel_size=1)
        self.conv15_se2_activation = hsigmoid()
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)

        self.conv15_linear = nn.Conv2d(480, 80, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv15_linear_bn = nn.BatchNorm2d(80)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)

        # for 14 [3,960,160,False,'hard_swish',1]
        # use_se
        self.conv16_expand = nn.Conv2d(80, 480, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv16_expand_bn = nn.BatchNorm2d(480)
        self.conv16_expand_bn_activation = hswish()

        self.conv16_depthwise = nn.Conv2d(480, 480, kernel_size=5, stride=1, padding=2, groups=480, bias=False)
        self.conv16_depthwise_bn = nn.BatchNorm2d(480)
        self.conv16_depthwise_bn_activation = hswish()

        # se
        self.conv16_se_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv16_se1 = nn.Conv2d(480, 120, kernel_size=1)
        self.conv16_se1_activation = nn.ReLU(inplace=True)
        self.conv16_se2 = nn.Conv2d(120, 480, kernel_size=1)
        self.conv16_se2_activation = hsigmoid()
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)

        self.conv16_linear = nn.Conv2d(480, 80, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv16_linear_bn = nn.BatchNorm2d(80)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)

        # last
        self.conv_last_expand = nn.Conv2d(80, 480, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv_last_expand_bn = nn.BatchNorm2d(480)
        self.conv_last_expand_bn_activation = hswish()

        self.max_pool = nn.MaxPool2d(2, stride=2)  # 括号内第一个参数是:窗口的大小,第二个是移动的步长距离

        self.rnn = nn.Sequential(
            BidirectionalLSTM(480, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv1_bn(output)
        output = self.conv1_bn_activation(output)

        # 0
        input1 = output
        output = self.conv2_expand(output)
        output = self.conv2_expand_bn(output)
        output = self.conv2_expand_bn_activation(output)

        output = self.conv2_depthwise(output)
        output = self.conv2_depthwise_bn(output)
        output = self.conv2_depthwise_bn_activation(output)

        output = self.conv2_linear(output)
        output = self.conv2_linear_bn(output)
        # self.conv2_linear_elementwise = fluid.layers.elementwise_add(x=input = A, y = conv2, act = None)  # 逐个元素相加
        output = input1 + output

        # 1
        output = self.conv3_expand(output)
        output = self.conv3_expand_bn(output)
        output = self.conv3_expand_bn_activation(output)

        output = self.conv3_depthwise(output)
        output = self.conv3_depthwise_bn(output)
        output = self.conv3_depthwise_bn_activation(output)

        output = self.conv3_linear(output)
        output = self.conv3_linear_bn(output)

        # for 2
        input1 = output
        output = self.conv4_expand(output)
        output = self.conv4_expand_bn(output)
        output = self.conv4_expand_bn_activation(output)

        output = self.conv4_depthwise(output)
        output = self.conv4_depthwise_bn(output)
        output = self.conv4_depthwise_bn_activation(output)

        output = self.conv4_linear(output)
        output = self.conv4_linear_bn(output)
        output = input1 + output

        # for 3 [5,72,40,True,'relu',(2,1)]
        # use_se
        output = self.conv5_expand(output)
        output = self.conv5_expand_bn(output)
        output = self.conv5_expand_bn_activation(output)

        output = self.conv5_depthwise(output)
        output = self.conv5_depthwise_bn(output)
        output = self.conv5_depthwise_bn_activation(output)

        # se
        output = self.conv5_se_avg(output)
        output = self.conv5_se1(output)
        output = self.conv5_se1_activation(output)
        output = self.conv5_se2(output)
        output = self.conv5_se2_activation(output)

        output = self.conv5_linear(output)
        output = self.conv5_linear_bn(output)

        # for 4 [5,120,40,True,'relu',1]
        # use_se
        input1 = output
        output = self.conv6_expand(output)
        output = self.conv6_expand_bn(output)
        output = self.conv6_expand_bn_activation(output)

        output = self.conv6_depthwise(output)
        output = self.conv6_depthwise_bn(output)
        output = self.conv6_depthwise_bn_activation(output)
        # se
        input = output
        output = self.conv6_se_avg(output)
        output = self.conv6_se1(output)
        output = self.conv6_se1_activation(output)
        output = self.conv6_se2(output)
        output = self.conv6_se2_activation(output)
        output = output * input

        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)


        output = self.conv6_linear(output)
        output = self.conv6_linear_bn(output)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        output = output + input1

        # for 5 [5,120,40,True,'relu',1]
        # use_se
        input1 = output
        output = self.conv7_expand(output)
        output = self.conv7_expand_bn(output)
        output = self.conv7_expand_bn_activation(output)

        output = self.conv7_depthwise(output)
        output = self.conv7_depthwise_bn(output)
        output = self.conv7_depthwise_bn_activation(output)

        # se
        input = output
        output = self.conv7_se_avg(output)
        output = self.conv7_se1(output)
        output = self.conv7_se1_activation(output)
        output = self.conv7_se2(output)
        output = self.conv7_se2_activation(output)
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)
        output = output * input

        output = self.conv7_linear(output)
        output = self.conv7_linear_bn(output)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        output = output + input1

        # for 6 [3,240,80,False,'hard_swish',1]
        output = self.conv8_expand(output)
        output = self.conv8_expand_bn(output)
        output = self.conv8_expand_bn_activation(output)

        output = self.conv8_depthwise(output)
        output = self.conv8_depthwise_bn(output)
        output = self.conv8_depthwise_bn_activation(output)

        output = self.conv8_linear(output)
        output = self.conv8_linear_bn(output)

        # for 7 [3,240,80,False,'hard_swish',1]
        input1 = output
        output = self.conv9_expand(output)
        output = self.conv9_expand_bn(output)
        output = self.conv9_expand_bn_activation(output)

        output = self.conv9_depthwise(output)
        output = self.conv9_depthwise_bn(output)
        output = self.conv9_depthwise_bn_activation(output)

        output = self.conv9_linear(output)
        output = self.conv9_linear_bn(output)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        output = output + input1

        # for 8 [3,184,80,False,'hard_swish',1]
        input1 = output
        output = self.conv10_expand(output)
        output = self.conv10_expand_bn(output)
        output = self.conv10_expand_bn_activation(output)

        output = self.conv10_depthwise(output)
        output = self.conv10_depthwise_bn(output)
        output = self.conv10_depthwise_bn_activation(output)

        output = self.conv10_linear(output)
        output = self.conv10_linear_bn(output)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        output = input1 + output


        # for 9 [3,184,80,False,'hard_swish',1]
        input1 = output
        output = self.conv11_expand(output)
        output = self.conv11_expand_bn(output)
        output = self.conv11_expand_bn_activation(output)

        output = self.conv11_depthwise(output)
        output = self.conv11_depthwise_bn(output)
        output = self.conv11_depthwise_bn_activation(output)

        output = self.conv11_linear(output)
        output = self.conv11_linear_bn(output)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        output = input1 + output

        # for 10 [3,480,112,False,'hard_swish',1]
        # use_se
        output = self.conv12_expand(output)
        output = self.conv12_expand_bn(output)
        output = self.conv12_expand_bn_activation(output)

        output = self.conv12_depthwise(output)
        output = self.conv12_depthwise_bn(output)
        output = self.conv12_depthwise_bn_activation(output)

        # se
        input = output
        output = self.conv12_se_avg(output)
        output = self.conv12_se1(output)
        output = self.conv12_se1_activation(output)
        output = self.conv12_se2(output)
        output = self.conv12_se2_activation(output)
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)
        output = input*output

        output = self.conv12_linear(output)
        output = self.conv12_linear_bn(output)

        # for 11 [3,672,112,False,'hard_swish',1]
        # use_se
        input1 = output
        output = self.conv13_expand(output)
        output = self.conv13_expand_bn(output)
        output = self.conv13_expand_bn_activation(output)

        output = self.conv13_depthwise(output)
        output = self.conv13_depthwise_bn(output)
        output = self.conv13_depthwise_bn_activation(output)

        # se
        input = output
        output = self.conv13_se_avg(output)
        output = self.conv13_se1(output)
        output = self.conv13_se1_activation(output)
        output = self.conv13_se2(output)
        output = self.conv13_se2_activation(output)
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)
        output = input*output

        output = self.conv13_linear(output)
        output = self.conv13_linear_bn(output)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        output = input1 + output

        # for 12 [3,672,160,False,'hard_swish',1]
        # use_se
        output = self.conv14_expand(output)
        output = self.conv14_expand_bn(output)
        output = self.conv14_expand_bn_activation(output)

        output = self.conv14_depthwise(output)
        output = self.conv14_depthwise_bn(output)
        output = self.conv14_depthwise_bn_activation(output)

        # se
        input = output
        output = self.conv14_se_avg(output)
        output = self.conv14_se1(output)
        output = self.conv14_se1_activation(output)
        output = self.conv14_se2(output)
        output = self.conv14_se2_activation(output)
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)
        output = output * input

        output = self.conv14_linear(output)
        output = self.conv14_linear_bn(output)

        # for 13 [3,960,160,False,'hard_swish',1]
        # use_se
        input1 = output
        output = self.conv15_expand(output)
        output = self.conv15_expand_bn(output)
        output = self.conv15_expand_bn_activation(output)

        self.conv15_depthwise(output)
        self.conv15_depthwise_bn(output)
        self.conv15_depthwise_bn_activation(output)

        # se
        input = output
        output = self.conv15_se_avg(output)
        output = self.conv15_se1(output)
        output = self.conv15_se1_activation(output)
        output = self.conv15_se2(output)
        output = self.conv15_se2_activation(output)
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)
        output = output * input

        output = self.conv15_linear(output)
        output = self.conv15_linear_bn(output)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        output = input1 + output

        # for 14 [3,960,160,False,'hard_swish',1]
        # use_se
        input1 = output
        output = self.conv16_expand(output)
        output = self.conv16_expand_bn(output)
        output = self.conv16_expand_bn_activation(output)

        output = self.conv16_depthwise(output)
        output = self.conv16_depthwise_bn(output)
        output = self.conv16_depthwise_bn_activation(output)

        # se
        input = output
        output = self.conv16_se_avg(output)
        output = self.conv16_se1(output)
        output = self.conv16_se1_activation(output)
        output = self.conv16_se2(output)
        output = self.conv16_se2_activation(output)
        # scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)
        output = output * input

        output = self.conv16_linear(output)
        output = self.conv16_linear_bn(output)
        # fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        output = input1 + output

        # last
        output = self.conv_last_expand(output)
        output = self.conv_last_expand_bn(output)
        output = self.conv_last_expand_bn_activation(output)

        output = self.max_pool(output)

        b, c, h, w = output.size()
        assert h == 1, "the height of conv must be 1"
        conv = output.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)  # 294*1*5991
        # output = self.rnn(conv)

        return output





