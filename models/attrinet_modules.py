import torch
import torch.nn as nn
import numpy as np
import math

"""
Task switching related code adapted from: https://github.com/GuoleiSun/TSNs/blob/main/model.py
Generator and Discriminator code adapted from: https://github.com/yunjey/stargan/blob/master/model.py
"""

class ScaleW:
    """
    Constructor: name - name of attribute to be scaled.
    """

    def __init__(self, name):
        self.name = name

    def scale(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        """
        Apply runtime scaling to specific module.
        """
        hook = ScaleW(name)
        weight = getattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        del module._parameters[name]
        module.register_forward_pre_hook(hook)

    def __call__(self, module, whatever):
        weight = self.scale(module)
        setattr(module, self.name, weight)


# Quick apply for scaled weight.
def quick_scale(module, name='weight'):
    ScaleW.apply(module, name)
    return module

class SLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        linear = nn.Linear(dim_in, dim_out)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        self.linear = quick_scale(linear)

    def forward(self, x):
        return self.linear(x)


# Normalization on every element of input vector.
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class FC_A(nn.Module):
    """
    Learned affine transform A, this module is used to transform midiate vector w into a style vector.
    """

    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = SLinear(dim_latent, n_channel * 2)
        # "the biases associated with ys that we initialize to one"
        self.transform.linear.bias.data[:n_channel] = 1
        self.transform.linear.bias.data[n_channel:] = 0

    def forward(self, w):
        # Gain scale factor and bias with:
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style


# AdaIn (AdaptiveInstanceNorm)
class AdaIn(nn.Module):
    """
    Adaptive instance normalization.
    """

    def __init__(self, n_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)   ## default

    def forward(self, image, style):
        factor, bias = style.chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias
        return result


class ResidualBlock_with_Ada(nn.Module):
    """
    Residual Block with adaptive instance normalization.
    """

    def __init__(self, dim_in, dim_out, act_func, dim_latent):
        super(ResidualBlock_with_Ada, self).__init__()

        # Style generators
        self.style1 = FC_A(dim_latent, dim_out)
        self.style2 = FC_A(dim_latent, dim_out)
        # AdaIn
        self.adain1 = AdaIn(dim_out)
        self.adain2 = AdaIn(dim_out)
        self.act_func = act_func
        # Convolutional layers
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, latent_w):
        result = self.conv1(x)
        result = self.adain1(result, self.style1(latent_w))
        result = self.act_func(result)
        result = self.conv2(result)
        result = self.adain2(result, self.style2(latent_w))
        return x + result


class Intermediate_Generator(nn.Module):
    """
    A mapping consists of multiple fully connected layers.
    Used to map the input to an intermediate latent space W.
    """

    def __init__(self, n_fc, dim_latent):
        super().__init__()
        layers = [PixelNorm()]
        for i in range(n_fc):
            layers.append(SLinear(dim_latent, dim_latent))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)

    def forward(self, latent_z):
        latent_w = self.mapping(latent_z)
        return latent_w



class convrelu(nn.Module):
    """
    This is the general class of style-based convolutional blocks
    """
    def __init__(self, in_channel, out_channel, kernel, stride, padding, dim_latent, bias=True, act_func=None):
        super().__init__()
        # Style generators
        self.style1 = FC_A(dim_latent, out_channel)
        # AdaIn
        self.adain = AdaIn(out_channel)
        self.act_func = act_func
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel, stride=stride, padding=padding, bias=bias)

    def forward(self, previous_result, latent_w):
        result = self.conv1(previous_result)
        result = self.adain(result, self.style1(latent_w))
        result = self.act_func(result)
        return result

class transconvrelu(nn.Module):
    """
    This is the general class of style-based convolutional blocks
    """

    def __init__(self, in_channel, out_channel, kernel, stride, padding, dim_latent, bias=True, act_func=None):
        super().__init__()
        # Style generators
        self.style1 = FC_A(dim_latent, out_channel)
        # AdaIn
        self.adain = AdaIn(out_channel)
        self.act_func = act_func
        # Convolutional layers
        self.transconv1 = nn.ConvTranspose2d(in_channel, out_channel, kernel, stride=stride, padding=padding, bias=bias)

    def forward(self, previous_result, latent_w):

        result = self.transconv1(previous_result)
        result = self.adain(result, self.style1(latent_w))
        result = self.act_func(result)
        return result





class Generator_with_Ada(nn.Module):
    """
    Generator network.
    """
    def __init__(self, num_classes, img_size, num_masks, act_func, n_fc=8, dim_latent=512, conv_dim=64, in_channels=1, repeat_num=6):
        super(Generator_with_Ada, self).__init__()

        self.num_classes = num_classes
        self.img_size = img_size
        self.repeat_num = repeat_num
        self.dim_latent = dim_latent

        if act_func == "relu":
            activation_func = nn.ReLU(inplace=True)  # original starGAN model use relu
        if act_func == "leakyrelu":
            activation_func = nn.LeakyReLU(inplace=True)  # original task switching model use leakyrelu

        # Task embedding layers.
        self.fcs = Intermediate_Generator(n_fc, dim_latent)

        self.layer0 = convrelu(in_channels, conv_dim, kernel=7, stride=1, padding=3, dim_latent=dim_latent, bias=False, act_func= activation_func)
        # Down-sampling layers.
        curr_dim = conv_dim
        self.down1 = convrelu(curr_dim, curr_dim * 2, kernel=4, stride=2, padding=1, dim_latent=dim_latent, bias=False, act_func= activation_func)
        curr_dim = curr_dim * 2
        self.down2 = convrelu(curr_dim, curr_dim * 2, kernel=4, stride=2, padding=1, dim_latent=dim_latent, bias=False, act_func= activation_func)
        curr_dim = curr_dim * 2

        # Bottleneck layers.
        self.bn_layer0 = ResidualBlock_with_Ada(dim_in=curr_dim, dim_out=curr_dim, act_func=activation_func,
                                                dim_latent=dim_latent)
        self.bn_layer1 = ResidualBlock_with_Ada(dim_in=curr_dim, dim_out=curr_dim, act_func=activation_func,
                                                dim_latent=dim_latent)
        self.bn_layer2 = ResidualBlock_with_Ada(dim_in=curr_dim, dim_out=curr_dim, act_func=activation_func,
                                                dim_latent=dim_latent)
        self.bn_layer3 = ResidualBlock_with_Ada(dim_in=curr_dim, dim_out=curr_dim, act_func=activation_func,
                                                dim_latent=dim_latent)
        self.bn_layer4 = ResidualBlock_with_Ada(dim_in=curr_dim, dim_out=curr_dim, act_func=activation_func,
                                                dim_latent=dim_latent)
        self.bn_layer5 = ResidualBlock_with_Ada(dim_in=curr_dim, dim_out=curr_dim, act_func=activation_func,
                                                dim_latent=dim_latent)

        self.bn_out_channel = curr_dim
        # Up-sampling layers.
        self.up1 = transconvrelu(curr_dim, curr_dim // 2, kernel=4, stride=2, padding=1, dim_latent=dim_latent, bias=False, act_func= activation_func)
        curr_dim = curr_dim // 2
        self.up2 = transconvrelu(curr_dim, curr_dim // 2, kernel=4, stride=2, padding=1, dim_latent=dim_latent, bias=False, act_func=activation_func)
        curr_dim = curr_dim // 2
        self.conv1 = nn.Conv2d(curr_dim, num_masks, kernel_size=7, stride=1, padding=3, bias=False)




    def forward(self, x, latent_z):
        """
        Input is the input image and latent_z is the input code for the corresponding task.
        """

        # Create the intermediate task embedding vector.
        if type(latent_z) != type([]):
            # print('You should use list to package your latent_z')
            latent_z = [latent_z]
        # latent_w as well as current_latent is the intermediate vector
        latent_w = [self.fcs(latent) for latent in latent_z]
        current_latent1 = latent_w
        current_latent = current_latent1[0]

        out = self.layer0(x, current_latent)
        out = self.down1(out, current_latent)
        out = self.down2(out, current_latent)

        out = self.bn_layer0(out,current_latent)
        out = self.bn_layer1(out, current_latent)
        out = self.bn_layer2(out, current_latent)
        out = self.bn_layer3(out, current_latent)
        out = self.bn_layer4(out, current_latent)
        out = self.bn_layer5(out, current_latent)
        out = self.up1(out, current_latent)
        out = self.up2(out, current_latent)
        out = self.conv1(out)

        # Create the dest image and mask.
        y = x + out
        y = torch.tanh(y)
        m = y - x
        return y, m


class Discriminator_with_Ada(nn.Module):
    """
    Discriminator network.
    """

    def __init__(self, act_func, conv_dim = 64, n_fc=8, dim_latent=512, repeat_num=6):
        super(Discriminator_with_Ada, self).__init__()

        if act_func == "relu":
            activation_func = nn.ReLU()  # original starGAN model use relu
        if act_func == "leakyrelu":
            activation_func = nn.LeakyReLU()  # original task switching model use leakyrelu

        # Task embedding layers.
        self.fcs = Intermediate_Generator(n_fc, dim_latent)

        self.layer0 = convrelu(1, conv_dim, kernel=4, stride=1, padding=1, dim_latent=dim_latent, bias=False, act_func=activation_func)
        curr_dim = conv_dim
        self.layer1 = convrelu(curr_dim, curr_dim * 2, kernel=4, stride=2, padding=1,dim_latent=dim_latent, bias=False, act_func=activation_func)
        curr_dim = curr_dim * 2
        self.layer2 = convrelu(curr_dim, curr_dim * 2, kernel=4, stride=2, padding=1, dim_latent=dim_latent, bias=False, act_func=activation_func)
        curr_dim = curr_dim * 2
        self.layer3 = convrelu(curr_dim, curr_dim * 2, kernel=4, stride=2, padding=1, dim_latent=dim_latent, bias=False, act_func=activation_func)
        curr_dim = curr_dim * 2
        self.layer4 = convrelu(curr_dim, curr_dim * 2, kernel=4, stride=2, padding=1, dim_latent=dim_latent, bias=False, act_func=activation_func)
        curr_dim = curr_dim * 2
        self.layer5 = convrelu(curr_dim, curr_dim * 2, kernel=4, stride=2, padding=1, dim_latent=dim_latent, bias=False, act_func=activation_func)
        curr_dim = curr_dim * 2
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x, latent_z):

        # Create the intermediate task embedding vector.
        if type(latent_z) != type([]):
            # print('You should use list to package your latent_z')
            latent_z = [latent_z]
        latent_w = [self.fcs(latent) for latent in latent_z]
        current_latent1 = latent_w
        current_latent = current_latent1[0]

        x = self.layer0(x, current_latent)
        x = self.layer1(x, current_latent)
        x = self.layer2(x, current_latent)
        x = self.layer3(x, current_latent)
        x = self.layer4(x, current_latent)
        h = self.layer5(x, current_latent)
        out_src = self.conv1(h)

        return out_src.view(out_src.size(0), -1).mean(1)





class CenterLoss(nn.Module):
    """
    Center loss.
    implementation adapt from https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py
    """

    def __init__(self, num_classes=2, feat_dim=102400, device=None):
        """
        Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))


    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss









