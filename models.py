import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import math,time


def DeconvBlock(in_channels,out_channels,kernel_size):
  return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,
                stride=2,padding = (kernel_size - 1) // 2,output_padding = kernel_size%2,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
def ConvBlock(in_channels,out_channels,kernel_size,stride,padding):
  return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

def ConvReLU6Block(in_channels,out_channels,kernel_size,stride,padding):
  return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
def DWConvBlock(in_channels,out_channels,kernel_size,stride,padding = None):
  if padding == None:
    padding = (kernel_size - 1) // 2
  return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False,groups=in_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

def DWConvReLU6Block(in_channels,out_channels,kernel_size,stride,padding = None):
  if padding == None:
    padding = (kernel_size - 1) // 2
  return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False,groups=in_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
def weights_init(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        

class MobileNet_Encoder(nn.Module):
    def __init__(self):
      super(MobileNet_Encoder, self).__init__()
      self.enc_layer1 = ConvReLU6Block(3,32,3,2,1)
      self.enc_layer2= nn.Sequential(DWConvReLU6Block(32,32,3,1,1),ConvReLU6Block(32,64,1,1,0))
      self.enc_layer3= nn.Sequential(DWConvReLU6Block(64,64,3,2,1),ConvReLU6Block(64,128,1,1,0))
      self.enc_layer4= nn.Sequential(DWConvReLU6Block(128,128,3,1,1),ConvReLU6Block(128,128,1,1,0))
      self.enc_layer5 = nn.Sequential(DWConvReLU6Block(128,128,3,2,1),ConvReLU6Block(128,256,1,1,0))
      self.enc_layer6 = nn.Sequential(DWConvReLU6Block(256,256,3,1,1),ConvReLU6Block(256,256,1,1,0))
      self.enc_layer7 = nn.Sequential(DWConvReLU6Block(256,256,3,2,1),ConvReLU6Block(256,512,1,1,0))
      self.enc_layer8 = nn.Sequential(DWConvReLU6Block(512,512,3,1,1),ConvReLU6Block(512,512,1,1,0))
      self.enc_layer9 = nn.Sequential(DWConvReLU6Block(512,512,3,1,1),ConvReLU6Block(512,512,1,1,0))
      self.enc_layer10 = nn.Sequential(DWConvReLU6Block(512,512,3,1,1),ConvReLU6Block(512,512,1,1,0))
      self.enc_layer11 = nn.Sequential(DWConvReLU6Block(512,512,3,1,1),ConvReLU6Block(512,512,1,1,0))
      self.enc_layer12 = nn.Sequential(DWConvReLU6Block(512,512,3,1,1),ConvReLU6Block(512,512,1,1,0))
      self.enc_layer13 =  nn.Sequential(DWConvReLU6Block(512,512,3,2,1),ConvReLU6Block(512,1024,1,1,0))
      self.enc_layer14 =  nn.Sequential(DWConvReLU6Block(1024,1024,3,1,1),ConvReLU6Block(1024,1024,1,1,0))
      self.enc_layer15= nn.AvgPool2d(7)
      self.enc_output = nn.Linear(1024, 1000)

    def forward(self, x):
      x=self.enc_layer1(x)
      x=self.enc_layer2(x)
      x=self.enc_layer3(x)
      x=self.enc_layer4(x)
      x=self.enc_layer5(x)
      x=self.enc_layer6(x)
      x=self.enc_layer7(x)
      x=self.enc_layer8(x)
      x=self.enc_layer9(x)
      x=self.enc_layer10(x)
      x=self.enc_layer11(x)
      x=self.enc_layer12(x)
      x=self.enc_layer13(x)
      x=self.enc_layer14(x)
      x=self.enc_layer15(x)
      return self.enc_output(x)
    
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)



class MobileNetV2_Encoder(nn.Module):
    def __init__(self):
        super(MobileNetV2_Encoder, self).__init__()
        self.enc_layer0 = ConvReLU6Block(3,32,3,2,1)
        self.enc_layer1= InvertedResidualBlock(32,16,1,1)
        self.enc_layer2= InvertedResidualBlock(16,24,2,6)
        self.enc_layer3= InvertedResidualBlock(24,24,1,6) 
        self.enc_layer4 = InvertedResidualBlock(24,32,2,6)
        self.enc_layer5 = InvertedResidualBlock(32,32,1,6)
        self.enc_layer6 = InvertedResidualBlock(32,32,1,6)
        self.enc_layer7 = InvertedResidualBlock(32,64,2,6)
        self.enc_layer8 = InvertedResidualBlock(64,64,1,6)
        self.enc_layer9 = InvertedResidualBlock(64,64,1,6)
        self.enc_layer10 = InvertedResidualBlock(64,64,1,6)
        self.enc_layer11 = InvertedResidualBlock(64,96,1,6)
        self.enc_layer12 =  InvertedResidualBlock(96,96,1,6)
        self.enc_layer13 =  InvertedResidualBlock(96,96,1,6)
        self.enc_layer14 = InvertedResidualBlock(96,160,2,6)
        self.enc_layer15 = InvertedResidualBlock(160,160,1,6)
        self.enc_layer16 = InvertedResidualBlock(160,160,1,6)
        self.enc_layer17 = InvertedResidualBlock(160,320,1,6)
        self.enc_layer18 = ConvReLU6Block(320,1280,1,1,0)
        self.enc_layer19 = nn.AvgPool2d(7)
        self.enc_output = nn.Linear(1280,1000)

    def forward(self, x):
        x=self.enc_layer0(x)
        x=self.enc_layer1(x)
        x=self.enc_layer2(x)
        x=self.enc_layer3(x)
        x=self.enc_layer4(x)
        x=self.enc_layer5(x)
        x=self.enc_layer6(x)
        x=self.enc_layer7(x)
        x=self.enc_layer8(x)
        x=self.enc_layer9(x)
        x=self.enc_layer10(x)
        x=self.enc_layer11(x)
        x=self.enc_layer12(x)
        x=self.enc_layer13(x)
        x=self.enc_layer14(x)
        x=self.enc_layer15(x)
        x=self.enc_layer16(x)
        x=self.enc_layer17(x)
        x=self.enc_layer18(x)
        x=self.enc_layer19(x)
        return self.enc_output(x)

    def forward(self, x):
        x=self.enc_layer0(x)
        x=self.enc_layer1(x)
        x=self.enc_layer2(x)
        x=self.enc_layer3(x)
        x=self.enc_layer4(x)
        x=self.enc_layer5(x)
        x=self.enc_layer6(x)
        x=self.enc_layer7(x)
        x=self.enc_layer8(x)
        x=self.enc_layer9(x)
        x=self.enc_layer10(x)
        x=self.enc_layer11(x)
        x=self.enc_layer12(x)
        x=self.enc_layer13(x)
        x=self.enc_layer14(x)
        x=self.enc_layer15(x)
        x=self.enc_layer16(x)
        x=self.enc_layer17(x)
        x=self.enc_layer18(x)
        x=self.enc_layer19(x)
        return self.enc_output(x)



class NNConv5_Decoder(nn.Module):
  def __init__(self, kernel_size, depthwise=True):
    super(NNConv5_Decoder, self).__init__()
    if (depthwise):
      self.conv1 = nn.Sequential(DWConvBlock(1024,1024,kernel_size,1),ConvBlock(1024,512,1,1,0))
      self.conv2 = nn.Sequential(DWConvBlock(512,512,kernel_size,1),ConvBlock(512,256,1,1,0))
      self.conv3 = nn.Sequential(DWConvBlock(256,256,kernel_size,1),ConvBlock(256,128,1,1,0))
      self.conv4 = nn.Sequential(DWConvBlock(128,128,kernel_size,1),ConvBlock(128,64,1,1,0))
      self.conv5 = nn.Sequential(DWConvBlock(64,64,kernel_size,1),ConvBlock(64,32,1,1,0))
    else:
      self.conv1 = ConvBlock(1024,512,kernel_size,1,(kernel_size - 1)//2)
      self.conv2 = ConvBlock(512,256,kernel_size,1,(kernel_size - 1)//2)
      self.conv3 = ConvBlock(256,128,kernel_size,1,(kernel_size - 1)//2)
      self.conv4 = ConvBlock(128,64,kernel_size,1,(kernel_size - 1)//2)
      self.conv5 = ConvBlock(64,32,kernel_size,1,(kernel_size - 1)//2)
    self.output = ConvBlock(32,1,1,1,0)
  def forward(self,x):
    x = F.interpolate(self.conv1(x), scale_factor=2, mode='nearest')
    x = F.interpolate(self.conv2(x), scale_factor=2, mode='nearest')
    x = F.interpolate(self.conv3(x), scale_factor=2, mode='nearest')
    x = F.interpolate(self.conv4(x), scale_factor=2, mode='nearest')
    x = F.interpolate(self.conv5(x), scale_factor=2, mode='nearest')
    return self.output(x)

class NNConv5_DecoderV2(nn.Module):
  def __init__(self, kernel_size, depthwise=True):
    super(NNConv5_DecoderV2, self).__init__()
    if (depthwise):
      self.conv1 = nn.Sequential(DWConvBlock(320,320,kernel_size,1),ConvBlock(320,96,1,1,0)) #14X14
      self.conv2 = nn.Sequential(DWConvBlock(96,96,kernel_size,1),ConvBlock(96,32,1,1,0)) #28 X 28
      self.conv3 = nn.Sequential(DWConvBlock(32,32,kernel_size,1),ConvBlock(32,24,1,1,0)) # 56X56
      self.conv4 = nn.Sequential(DWConvBlock(24,24,kernel_size,1),ConvBlock(24,16,1,1,0)) #112 X 112
      self.conv5 = nn.Sequential(DWConvBlock(16,16,kernel_size,1),ConvBlock(16,32,1,1,0)) #224 X 224

    self.output = ConvBlock(32,1,1,1,0)
  def forward(self,x):
    x = F.interpolate(self.conv1(x), scale_factor=2, mode='nearest')
    x = F.interpolate(self.conv2(x), scale_factor=2, mode='nearest')
    x = F.interpolate(self.conv3(x), scale_factor=2, mode='nearest')
    x = F.interpolate(self.conv4(x), scale_factor=2, mode='nearest')
    x = F.interpolate(self.conv5(x), scale_factor=2, mode='nearest')
    return self.output(x)


class FastDepth(nn.Module):
  def __init__(self,kernel_size=5):
    super(FastDepth,self).__init__()
    self.encoder= MobileNet_Encoder()
    self.decoder = NNConv5_Decoder(kernel_size)
  def forward(self,x):
    x=self.encoder.enc_layer1(x)
    x=self.encoder.enc_layer2(x)
    layer1= x
    x=self.encoder.enc_layer3(x)
    x=self.encoder.enc_layer4(x)
    layer2=x
    x=self.encoder.enc_layer5(x)
    x=self.encoder.enc_layer6(x)
    layer3=x
    x=self.encoder.enc_layer7(x)
    x=self.encoder.enc_layer8(x)
    x=self.encoder.enc_layer9(x)
    x=self.encoder.enc_layer10(x)
    x=self.encoder.enc_layer11(x)
    x=self.encoder.enc_layer12(x)
    x=self.encoder.enc_layer13(x)
    x=self.encoder.enc_layer14(x)
    x= F.interpolate(self.decoder.conv1(x), scale_factor=2, mode='nearest')
    x= F.interpolate(self.decoder.conv2(x), scale_factor=2, mode='nearest')
    x = x+layer3
    x= F.interpolate(self.decoder.conv3(x), scale_factor=2, mode='nearest') + layer2
    x= F.interpolate(self.decoder.conv4(x), scale_factor=2, mode='nearest') + layer1
    x= F.interpolate(self.decoder.conv5(x), scale_factor=2, mode='nearest')
    return self.decoder.output(x)


class FastDepthV2(nn.Module):
  def __init__(self,kernel_size=5):
    super(FastDepthV2,self).__init__()
    self.encoder= MobileNetV2_Encoder()
    self.decoder = NNConv5_DecoderV2(kernel_size)
  def forward(self,x):
    x=self.encoder.enc_layer0(x)
    x=self.encoder.enc_layer1(x)
    layer1= x
    x=self.encoder.enc_layer2(x)
    x=self.encoder.enc_layer3(x)
    layer2=x
    x=self.encoder.enc_layer4(x)
    x=self.encoder.enc_layer5(x)
    layer3=x
    x=self.encoder.enc_layer6(x)
    x=self.encoder.enc_layer7(x)
    x=self.encoder.enc_layer8(x)
    x=self.encoder.enc_layer9(x)
    x=self.encoder.enc_layer10(x)
    x=self.encoder.enc_layer11(x)
    x=self.encoder.enc_layer12(x)
    x=self.encoder.enc_layer13(x)
    x=self.encoder.enc_layer14(x)
    x=self.encoder.enc_layer15(x)
    x=self.encoder.enc_layer16(x)
    x=self.encoder.enc_layer17(x)
    x= F.interpolate(self.decoder.conv1(x), scale_factor=2, mode='nearest')
    x= F.interpolate(self.decoder.conv2(x), scale_factor=2, mode='nearest')
    x = x+layer3
    x= F.interpolate(self.decoder.conv3(x), scale_factor=2, mode='nearest')
    x = x+layer2
    x= F.interpolate(self.decoder.conv4(x), scale_factor=2, mode='nearest')
    x = x+layer1
    x= F.interpolate(self.decoder.conv5(x), scale_factor=2, mode='nearest')
    return self.decoder.output(x)