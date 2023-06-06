import os
import sys
sys.path.append(os.path.join(sys.path[0], "../utils/")) 
import torch 
import torch.nn as nn
from hawq.utils.quantization_utils.quant_modules import QuantAct, QuantLinear, QuantConv2d, QuantBnConv2d


class RN07(nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

    self.conv2d_0 = nn.Conv2d(3, 32, 1, 1, 0)
    self.batchnorm_0 = nn.BatchNorm2d(32)

    self.conv2d_1 = nn.Conv2d(32, 4, 1, 1, 0)
    self.batchnorm_1 = nn.BatchNorm2d(4)

    self.conv2d_2 = nn.Conv2d(4, 32, 1, 1, 0)
    self.batchnorm_2 = nn.BatchNorm2d(32)

    self.conv2d_3 = nn.Conv2d(32, 32, 4, 4, 0)
    self.batchnorm_3 = nn.BatchNorm2d(32)

    self.conv2d_4 = nn.Conv2d(32, 32, 1, 1, 0)
    self.batchnorm_4 = nn.BatchNorm2d(32)

    self.dense = nn.Linear(2048, 10)
    self.softmax = nn.Softmax(dim=0)

  def forward(self, x):
    x = self.batchnorm_0(self.conv2d_0(x))
    # q_activation_0
    x = self.batchnorm_1(self.conv2d_1(x))
    # q_activation_1
    x = self.batchnorm_2(self.conv2d_2(x))
    # q_activation_2
    x = self.batchnorm_3(self.conv2d_3(x))
    # q_activation_3
    x = self.batchnorm_4(self.conv2d_4(x))

    x = torch.flatten(x, 1)
    x = self.dense(x)
    x = self.softmax(x)
    return x

class Q_Base(nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def init_linear(self, model, name, precision):
    orig_layer = getattr(model, name)
    hawq_layer = QuantLinear(precision, precision)
    hawq_layer.set_param(orig_layer)
    setattr(self, name, hawq_layer)
  
  def init_conv2d(self, model, name, precision):
    orig_layer = getattr(model, name)
    hawq_layer = QuantConv2d(precision)
    hawq_layer.set_param(orig_layer)
    setattr(self, name, hawq_layer)

  def init_conv2d_batchnorm(self, model, conv_name, bn_name, precision):
    orig_conv_layer = getattr(model, conv_name)
    orig_bn_layer = getattr(model, bn_name)
    hawq_layer = QuantBnConv2d(precision)
    hawq_layer.set_param(orig_conv_layer, orig_bn_layer)
    setattr(self, conv_name, hawq_layer)

    
class Q_RN07(Q_Base):
  def __init__(self, model, w_p, b_p, a_p, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

    self.quant_input = QuantAct(a_p)

    self.init_conv2d(model, 'conv2d_0', w_p)
    self.batchnorm_0 = nn.BatchNorm2d(32)
    self.q_activation_0 = QuantAct(a_p)

    self.init_conv2d(model, 'conv2d_1', w_p)
    self.batchnorm_1 = nn.BatchNorm2d(4)
    self.q_activation_1 = QuantAct(a_p)

    self.init_conv2d(model, 'conv2d_2', w_p)
    self.batchnorm_2 = nn.BatchNorm2d(32)
    self.q_activation_2 = QuantAct(a_p)

    self.init_conv2d(model, 'conv2d_3', w_p)
    self.batchnorm_3 = nn.BatchNorm2d(32)
    self.q_activation_3 = QuantAct(a_p)

    self.init_conv2d(model, 'conv2d_4', w_p)
    self.batchnorm_4 = nn.BatchNorm2d(32)
    self.q_activation_4 = QuantAct(a_p)

    self.init_linear(model, 'dense', w_p)
    self.softmax = nn.Softmax(dim=0)
  
  def forward(self, x):
    x, p_sf = self.quant_input(x)

    x, w_sf = self.conv2d_0(x, p_sf)
    x       = self.batchnorm_0(x)
    x, p_sf = self.q_activation_0(x, p_sf, w_sf)

    x, w_sf = self.conv2d_1(x, p_sf)
    x       = self.batchnorm_1(x)
    x, p_sf = self.q_activation_1(x, p_sf, w_sf)

    x, w_sf = self.conv2d_2(x, p_sf)
    x       = self.batchnorm_2(x)
    x, p_sf = self.q_activation_2(x, p_sf, w_sf)

    x, w_sf = self.conv2d_3(x, p_sf)
    x       = self.batchnorm_3(x)
    x, p_sf = self.q_activation_3(x, p_sf, w_sf)

    x, w_sf = self.conv2d_4(x, p_sf)
    x       = self.batchnorm_4(x)
    x, p_sf = self.q_activation_4(x, p_sf, w_sf)

    x = x.reshape(-1,2048)
    x = self.dense(x, p_sf)
    x = self.softmax(x)
    return x


def get_rn07(args):
  model = RN07()
  if args.weight_precision < 32:
    model = Q_RN07(model, args.weight_precision, args.bias_precision, args.act_precision)
  return model
