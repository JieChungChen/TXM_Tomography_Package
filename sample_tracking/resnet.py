
import torch
import torch.nn as nn
from collections import OrderedDict


class Bottleneck(nn.Module):
    """
    Bottleneck 殘差塊，用於 ResNet50/101/152
    擴展因子為 4，即輸出通道數是中間層的 4 倍
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        # 1x1 卷積，降維
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 3x3 卷積，主要計算層
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 1x1 卷積，升維
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # 1x1 降維
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3 卷積
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1 升維
        out = self.conv3(out)
        out = self.bn3(out)

        # 如果需要下採樣，對 identity 進行調整
        if self.downsample is not None:
            identity = self.downsample(x)

        # 殘差連接
        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    """
    ResNet50 backbone for visual tracking
    """
    def __init__(self, output_layers=None, frozen_layers=None):
        """
        Args:
            output_layers: 要輸出的層名稱列表，例如 ['layer2', 'layer3']
            frozen_layers: 要凍結的層名稱列表，例如 ['conv1', 'bn1', 'layer1']
        """
        super(ResNet50, self).__init__()
        
        self.output_layers = output_layers if output_layers is not None else []
        self.frozen_layers = frozen_layers if frozen_layers is not None else []
        
        self.inplanes = 64
        
        # 初始卷積層 (7x7, stride=2)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 最大池化層 (3x3, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet50 的 4 個 stage
        # layer1: 3 個 bottleneck，輸出 256 通道
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        
        # layer2: 4 個 bottleneck，輸出 512 通道
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        
        # layer3: 6 個 bottleneck，輸出 1024 通道
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=1, dilation=2)
        
        
        # 初始化權重
        self._initialize_weights()
        
        # 凍結指定層
        self._freeze_layers()
    
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """
        構建 ResNet 的一個 stage
        
        Args:
            block: Bottleneck 類
            planes: 該 stage 中間層的通道數
            blocks: 該 stage 包含的 block 數量
            stride: 第一個 block 的步長
            dilation: 空洞卷積的擴張率
        """
        downsample = None
        
        # 如果步長不為1或輸入輸出維度不匹配，需要下採樣
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        # 第一個 block，可能需要下採樣
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        
        self.inplanes = planes * block.expansion
        
        # 後續的 blocks
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化網絡權重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _freeze_layers(self):
        """凍結指定的層，不更新其參數"""
        for name, module in self.named_children():
            if name in self.frozen_layers:
                for param in module.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x: 輸入張量 (B, 1, H, W)
        
        Returns:
            如果 output_layers 為空，返回最後一層的輸出
            否則返回字典，包含指定層的輸出
        """
        outputs = OrderedDict()
        
        # Stem (初始層)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if 'conv1' in self.output_layers:
            outputs['conv1'] = x
        
        x = self.maxpool(x)
        if 'maxpool' in self.output_layers:
            outputs['maxpool'] = x
        
        # Layer 1
        x = self.layer1(x)
        if 'layer1' in self.output_layers:
            outputs['layer1'] = x
        
        # Layer 2
        x = self.layer2(x)
        if 'layer2' in self.output_layers:
            outputs['layer2'] = x
        
        # Layer 3
        x = self.layer3(x)
        if 'layer3' in self.output_layers:
            outputs['layer3'] = x

        
        # 如果沒有指定輸出層，返回最後的特徵
        if len(outputs) == 0:
            return x
        print( self.output_layers)
        return outputs


def resnet50(pretrained=False, output_layers=None, frozen_layers=None):
    """
    構建 ResNet50 模型
    
    Args:
        pretrained: 是否加載預訓練權重
        output_layers: 要輸出的層
        frozen_layers: 要凍結的層
    
    Returns:
        ResNet50 模型
    """
    model = ResNet50(output_layers=output_layers, frozen_layers=frozen_layers)
    
    if pretrained:
        # 加載 ImageNet 預訓練權重
        import torchvision.models as models
        pretrained_dict = models.resnet50(pretrained=True).state_dict()
        model_dict = model.state_dict()
        
        # 只加載匹配的權重
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)} pretrained parameters")
    
    return model


# 使用範例
if __name__ == "__main__":
    # 創建模型，輸出 layer2 和 layer3
    model = resnet50(
        pretrained=False,
        output_layers=['layer3'],
    )
    
    # 測試前向傳播
    x = torch.randn(2, 1, 128, 128)
    outputs = model(x)
    
    print("輸出特徵:")
    for name, feat in outputs.items():
        print(f"{name}: {feat.shape}")
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n總參數量: {total_params:,}")
    print(f"可訓練參數量: {trainable_params:,}")