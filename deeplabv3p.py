import torch
import torch.nn as nn
from torch.nn import functional as F

# 
# 深度可分离卷积   
def separable_conv(in_channels, out_channels, stride):
    # depthwise conv，使用group conv，把每一个channel作为一组分别卷积，所以groups=in_channels
    depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels)
    # pointwise conv，使用1x1 conv，对分组卷积的结果进行卷积
    point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=1)
    return nn.Sequential(depth_conv, point_conv)

#封装3个3x3的separable conv, 将三次卷积后的结果与skip相加后的结果返回
#skip有两种情况：一种直接skip，一种需要sep conv后skip
class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, skip_conv=True):
        super(XceptionBlock, self).__init__()
        
        self.conv1 = nn.Sequential(
            separable_conv(in_channels, out_channels, stride[0]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            separable_conv(out_channels, out_channels, stride[1]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            separable_conv(out_channels, out_channels, stride[2]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.skip_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2)
        
    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.skip_conv:
            shortcut = self.skip_conv(shortcut)
        x = shortcut + x
        
        return x
    
class EntryFlow(nn.Module):
    def __init__(self):
        super(EntryFlow, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.block1 = XceptionBlock(64, 128, [1,1,2])
        self.block2 = XceptionBlock(128, 256, [1,1,2])
        self.block3 = XceptionBlock(256,728,[1,1,2])
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x))) # 1/2
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.block1(x) # 1/4
        
        # 保存下采样4倍的low level的feature map，包含了更多的位置信息，用于deeplabv3+的decoder的输入之一
        low_level_feature = x
        
        x = self.block2(x) # 1/8
        x = self.block3(x) # 1/16        
        
        return [x, low_level_feature]
    
    
class MiddleFlow(nn.Module):
    def __init__(self):
        super(MiddleFlow, self).__init__()        
        self.block = XceptionBlock(728,728,[1,1,1],skip_conv=False)
        
    def forward(self,x):
        for i in range(16):
            x = self.block(x)
        return x
    
class ExitFlow(nn.Module):
    def __init__(self):
        super(ExitFlow, self).__init__()
        self.conv1 = separable_conv(728, 728, 1)
        self.bn1 = nn.BatchNorm2d(728)
        self.relu1 = nn.ReLU()
        
        self.conv2 = separable_conv(728, 1024, 1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.relu2 = nn.ReLU()
        
        self.conv3 = separable_conv(1024, 1024, 2)
        self.bn3 = nn.BatchNorm2d(1024)
        self.relu3 = nn.ReLU()
        
        self.skip_conv = nn.Conv2d(728, 1024, 1, stride=2)
        
        self.conv4 = separable_conv(1024, 1536, 1)
        self.bn4 = nn.BatchNorm2d(1536)
        self.relu4 = nn.ReLU()
        
        self.conv5 = separable_conv(1536, 1536, 1)
        self.bn5 = nn.BatchNorm2d(1536)
        self.relu5 = nn.ReLU()
        
        self.conv6 = separable_conv(1536, 2048, 1)
        self.bn6 = nn.BatchNorm2d(2048)
        self.relu6 = nn.ReLU()
        
        
    def forward(self, x):
        shortcut = self.skip_conv(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x))) # 1/32
        #skip
        x = x + shortcut
        
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        
        return x

class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()
        
        #EntryFlow
        self.entryflow = EntryFlow()
        #MiddleFlow
        self.middleflow = MiddleFlow()
        #ExitFlow
        #self.exitflow = ExitFlow()
        
    def foward(self, x):
        res = self.entryflow(x)
        
        x, decoder_data = res[0], res[1:]
        
        x = self.middleflow(x)
        #x = self.exitflow(x)
        return [x, decoder_data]

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False), #膨胀卷积
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)
        
class ASPPPolling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPolling, self).__init__(
            nn.AdaptiveAvgPool2d(1), #全局平均池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        
    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    
#1x1conv + 3个3x3conv + global pooling
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rate):
        super(ASPP, self).__init__()
        out_channels = 256 #实验效果最好，来自论文
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        
        #rate: 对卷积核的膨胀率
        rate1, rate2, rate3 = tuple(atrous_rate)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPolling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        self.project = nn.Sequential(
            # concat 之后channels*5
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, di=1)        
        return self.project(res)
    

# 解码器配置
# encode_data：编码器输出
# decode_shortcut: 从backbone引出的分支, resize后与encode_data concat
class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()
        
        indim = 128 #Xception的第一个XceptionBlock之后的输出channel，作为1x1conv的in_channels，不知道对不对？
        dim1 = 48 #[1x1,48]来自原论文，实验得出的最好的效果是维数48
        dim2 = 256 #[3x3, 256]来自原论文
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(indim, dim1, 1),
            nn.BatchNorm2d(dim1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim1+dim2, dim2, 3),
            nn.BatchNorm2d(dim2),
            nn.ReLU(),
            nn.Conv2d(dim2, dim2, 3),
            nn.BatchNorm2d(dim2),
            nn.ReLU()
        )
        
        #需要再接一个1x1conv，调整维度到num_classes吗？
        self.class_conv = nn.Conv2d(dim2, out_channels, 1)
    
    def forward(self, x, shortcut):
        res = []
        # 对low level feature进行1x1conv降维
        decode_shortcut = self.conv1(shortcut)
        
        # 对encoder出来的data进行4倍上采样，采用双线性插值的方式
        encode_data = F.interplate(x, size=4, mode='bilinear', align_corners=False)
        
        # 对encode_data裁剪
        crop = self.center_crop(encode_data, decode_shortcut.shape[2:])        
        
        # 将shortcut和encoder的data拼接
        res = torch.cat((decode_shortcut,crop),dim=1)
        
        # 将拼接后的结果做3x3conv
        res = self.conv2(res)
        
        # 将结果上采样4倍输出
        res = F.interplate(res, size=4, mode='bilinear', align_corners=False)
        
        res = self.class_conv(res)
        
        return res
    
    #定义对feature map的剪裁函数
    def center_crop(self, layer, target_size):
        _,_,layer_height,layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:,:,diff_y:(diff_y + target_size[0]),diff_x:(diff_x + target_size[1])]
    
    
#编码器配置：backbone + aspp
#backbone，可选Xception或者ResNet101
#采用ASPP架构，pooling + 1x1_conv + 三个不同尺度的空洞卷积并行, concat后1x1conv
class Encoder(nn.Module):
    def __init__(self, backbone):
        super(Encoder, self).__init__()
        in_channels = 1024 #xception 1/16 之后，再次stride之前
        self.aspp = ASPP(in_channels, [6, 12, 18])
        if backbone == 'xception':
            self.backbone = Xception()
        
    def forward(self, x):
        res = self.backbone(x)
        x, decoder_data = res[0], res[1:]
        x = self.aspp(x)
        return [x, decoder_data]

    
class DeepLabV3p(nn.Module):
    def __init__(self, backbone, num_classes):
        super(DeepLabV3p, self).__init__()
        self.encoder = Encoder(backbone)
        self.decoder = Decoder(num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        res = self.encoder(x)
        x, decoder_data = self.decoder(res[0],res[1:])
        x = self.decoder(x, decoder_data)
        return x