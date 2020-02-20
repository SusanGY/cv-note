import torch
import torch.nn as nn

#封装2个卷积
def double_conv(in_channels, out_channels, batch_norm=None):
    #新建一个空的列表，按顺序装入模块
    block = []
    
    #第一次卷积，后跟ReLU激活函数
    #没有设padding=1，上课时ppt的图上显示卷积后图片是变小的
    block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3))    
    block.append(nn.ReLU(inplace=True))
    
    #检测是否有BN层
    if batch_norm is not None:
        block.append(nn.BatchNorm2d(out_channels))
        
    #第二次卷积，后跟ReLU激活函数
    block.append(nn.Conv2d(out_channels, out_channels, kernel_size=3))
    block.append(nn.ReLU(inplace=True))
    
    #检测是否有BN层
    if batch_norm is not None:
        block.append(nn.BatchNorm2d(out_channels))
        
    #返回序列化的模块
    return nn.Sequential(*block)

#下采样：max_pool + 2个卷积
class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm):
        super(UNetDownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = double_conv(in_channels, out_channels, batch_norm)
        
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x

#上采样：上采样(upconv或upsample) + resize并与shortcut concat + 2个卷积
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm, up_mode):
        super(UNetUpBlock, self).__init__()
        
        if up_mode == 'upconv': #采用转置卷积
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif up_mode == 'upsample': #采用双线性插值
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),#把feature map根据上采样率插值出更大的feature map，channel数不会改变
                nn.Conv2d(in_channels, out_channels, kernel_size=1) #调整channel
            )
        
        self.conv = double_conv(in_channels, out_channels, batch_norm)
    
    def forward(self, x, shortcut):
        # 对x上采样
        up = self.up(x)
        
        # 对shortcup裁剪
        crop = self.center_crop(shortcut, up.shape[2:])
        
        # 拼接x和shortcut
        out = torch.cat([up, crop],1)
        
        # 卷积
        out = self.conv(out)
        return out
    
    #定义对feature map的剪裁函数
    def center_crop(self, layer, target_size):
        _,_,layer_height,layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:,:,diff_y:(diff_y + target_size[0]),diff_x:(diff_x + target_size[1])]

# 输入通道数、输出通道数 都是根据上课时ppt的图上给出的数字进行设置的
class UNet(nn.Module):
    def __init__(self, num_classes=2, batch_norm=False, up_mode='upconv'):
        super(UNet, self).__init__()
        self.batch_norm = batch_norm
        self.up_mode = up_mode
        
        #第一层单独处理
        self.first = double_conv(1,64, batch_norm)
        
        #四次下采样
        self.down1 = UNetDownBlock(64, 128, batch_norm)
        self.down2 = UNetDownBlock(128, 256, batch_norm)
        self.down3 = UNetDownBlock(256, 512, batch_norm)
        self.down4 = UNetDownBlock(512, 1024, batch_norm)
        
        #四次上采样
        self.up4 = UNetUpBlock(1024, 512, batch_norm, up_mode)
        self.up3 = UNetUpBlock(512, 256, batch_norm, up_mode)
        self.up2 = UNetUpBlock(256, 128, batch_norm, up_mode)
        self.up1 = UNetUpBlock(128, 64, batch_norm, up_mode)
        
        #最后用1x1卷积调整维度
        self.last = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, x):
        shortcuts = []
        
        x = self.first(x)
        
        shortcuts.append(x)
        x = self.down1(x)
        
        shortcuts.append(x)
        x = self.down2(x)
        
        shortcuts.append(x)
        x = self.down3(x)
        
        shortcuts.append(x)
        x = self.down4(x)
        
        x = self.up4(x, shortcuts[3])
        x = self.up3(x, shortcuts[2])
        x = self.up2(x, shortcuts[1])
        x = self.up1(x, shortcuts[0])
        
        x = self.last(x)
        
        return x  


# 测试代码
# x = torch.randn((1,1,572,572))
# unet = UNet()
# unet.eval()
# y_unet = unet(x)

# y_unet.size()
# Output：torch.Size([1, 2, 388, 388])
