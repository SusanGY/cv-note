import torch
import torch.nn as nn
import numpy as np

# 从VGG16改写，将VGG16的全连接层换为卷积层，另外增加skip
# pool5上采样2倍后与pool4相加(add),得到的结果再次上采样2倍与pool3 add，得到的结果直接上采样8倍，得到spatial的heatmap
class FCN8s(nn.Module):
    def __init__(self, n_class=21): # n_class数值来自于论文
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100) # Output = (input - kerner_size + 2*padding)/stride + 1; output >= 7; fc6之前得到的feature map至少是7x7
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/2; 一次MaxPooling操作后，spatial缩小为原来的1/2; ceil_mode=True,表示不足square_size的边界信息保留
        
        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/4
        
        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/8
        
        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/16
        
        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/32
        
        # fc6 将VGG16的全连接层替换为卷积层
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        
        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        
        # 1x1卷积调整channel
        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)
        
        # 转置卷积，stride=N,kernel_size=2N表示上采样N倍
        self.upscore2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTransepose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0]
                )
                m.weight.data.copy_(initial_weight)
        
    def forward(self, x):
        h = x
        h = self.relu1_1(conv1_1(h))
        h = self.relu1_2(conv1_2(h))
        h = self.pool1(h)
        
        h = self.relu2_1(conv2_1(h))
        h = self.relu2_2(conv2_2(h))
        h = self.pool2(h)
        
        h = self.relu3_1(conv3_1(h))
        h = self.relu3_2(conv3_2(h))
        h = self.relu3_3(conv3_3(h))
        h = self.pool3(h)
        pool3 = h # 1/8
        
        h = self.relu4_1(conv4_1(h))
        h = self.relu4_2(conv4_2(h))
        h = self.relu4_3(conv4_3(h))
        h = self.pool4(h)
        pool4 = h # 1/16
        
        h = self.relu5_1(conv5_1(h))
        h = self.relu5_2(conv5_2(h))
        h = self.relu5_3(conv5_3(h))
        h = self.pool5(h)
        pool5 = h
        
        h = self.relu6(fc6(h))
        h = self.drop6(h)
        
        h = self.relu7(fc7(h))
        h = self.drop7(h)
        
        # 1x1卷积调整channel
        h = self.score_fr(h)
        # 转置卷积上采样2倍，由1/32变为1/16
        h = self.upscore2(h)
        upscore2 = h # 1/16
        
        # 1x1卷积调整channel
        h = self.score_pool4(pool4)
        # crop调整spatial，因为add需要维度一致:spatial和channel都要一致
        h = h[:, :, 5:5+upscore2.size()[2], 5:5+upscore2.size()[3]]
        score_pool4 = h
        
        # add
        h = upscore2 + score_pool4
        # 上采样2倍，变为1/8
        h = self.upscore_pool4(h)
        upscore_pool4 = h
        
        h = self.score_pool3(pool3)
        h = h[:, :, 9+upscore_pool4.size[2], 9+upscore_pool4.size[3]]
        score_pool3 = h
        
        # add
        h = upscore_pool4 + score_pool3
        
        # 直接上采样8倍得到输出
        h = self.upscore8(h)
        h = h[:, :, 31:31+x.size()[2], 31:31+x.size()[3]].contiguous()
        
        return h