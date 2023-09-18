import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from repvgg import RepVGG


# 注意力模块

class AttentionMap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionMap, self).__init__()
        self.register_buffer('mask', torch.zeros([1, 1, 12, 12]))   # 在寄存器缓冲区定义一个掩码大小为1通道1维24*24大小，先缩小一半
        self.mask[0, 0, 2:-2, 2:-2] = 1  # 第一个通道，第一个维度，高度2到倒数2，宽度2到倒数2设置为1
        self.num_attentions = out_channels  # 注意力图数量等于输出通道数
        self.conv_extract = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)    # 从主干提取特征图
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):  # 输入以forward为主
        if self.num_attentions == 0:  # 注意力图数量为0则返回创建size大小的维度，里面元素全部填充为1
            return torch.ones([x.shape[0], 1, 1, 1], device=x.device)
        x = self.conv_extract(x)  # 3*3卷积
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)+1  # ELU函数是针对ReLU函数的一个改进型
        mask = F.interpolate(self.mask, (x.shape[2], x.shape[3]), mode='nearest')  # 采样
        return x*mask


# 浅层文本特征提取

class Texture_Enhance_v2(nn.Module):
    def __init__(self, num_features, num_attentions):
        super().__init__()
        self.output_features = num_features  # 输出文本特征数
        self.output_features_d = num_features  # 输出无文本特征数
        self.conv_extract = nn.Conv2d(num_features, num_features, 3, padding=1)  # 从主干提取特征图
        self.conv0 = nn.Conv2d(num_features * num_attentions, num_features * num_attentions, 5, padding=2,
                               groups=num_attentions)
        self.conv1 = nn.Conv2d(num_features * num_attentions, num_features * num_attentions, 3, padding=1,
                               groups=num_attentions)
        self.bn1 = nn.BatchNorm2d(num_features * num_attentions)
        self.conv2 = nn.Conv2d(num_features * 2 * num_attentions, num_features * num_attentions, 3, padding=1,
                               groups=num_attentions)
        self.bn2 = nn.BatchNorm2d(2 * num_features * num_attentions)
        self.conv3 = nn.Conv2d(num_features * 3 * num_attentions, num_features * num_attentions, 3, padding=1,
                               groups=num_attentions)
        self.bn3 = nn.BatchNorm2d(3 * num_features * num_attentions)
        self.conv_last = nn.Conv2d(num_features * 4 * num_attentions, num_features * num_attentions, 1,
                                   groups=num_attentions)
        self.bn4 = nn.BatchNorm2d(4 * num_features * num_attentions)
        self.bn_last = nn.BatchNorm2d(num_features * num_attentions)

        self.M = num_attentions

    def cat(self, a, b):  # 在给定维度上对输入的张量序列seq 进行连接操作
        B, C, H, W = a.shape
        c = torch.cat([a.reshape(B, self.M, -1, H, W), b.reshape(B, self.M, -1, H, W)], dim=2).reshape(B, -1, H, W)
        return c

    def forward(self, feature_maps, attention_maps=(1, 1)):
        B, N, H, W = feature_maps.shape
        if type(attention_maps) == tuple:
            attention_size = (int(H * attention_maps[0]), int(W * attention_maps[1]))
        else:
            attention_size = (attention_maps.shape[2], attention_maps.shape[3])
        feature_maps = self.conv_extract(feature_maps)  # 下采样
        feature_maps_d = F.adaptive_avg_pool2d(feature_maps, attention_size)  # 平均池化
        if feature_maps.size(2) > feature_maps_d.size(2):   # 纹理特征的高大于去纹理特征的高进该判段
            feature_maps = feature_maps - F.interpolate(feature_maps_d, (feature_maps.shape[2], feature_maps.shape[3]),
                                                        mode='nearest')
        attention_maps = (
            torch.tanh(F.interpolate(attention_maps.detach(), (H, W), mode='bilinear', align_corners=True))).unsqueeze(
            2) if type(attention_maps) != tuple else 1
        feature_maps = feature_maps.unsqueeze(1)  # 在指定的位置插入一个维度
        feature_maps = (feature_maps * attention_maps).reshape(B, -1, H, W)  # 融合
        feature_maps0 = self.conv0(feature_maps)
        feature_maps1 = self.conv1(F.relu(self.bn1(feature_maps0), inplace=True))
        feature_maps1_ = self.cat(feature_maps0, feature_maps1)
        feature_maps2 = self.conv2(F.relu(self.bn2(feature_maps1_), inplace=True))
        feature_maps2_ = self.cat(feature_maps1_, feature_maps2)
        feature_maps3 = self.conv3(F.relu(self.bn3(feature_maps2_), inplace=True))
        feature_maps3_ = self.cat(feature_maps2_, feature_maps3)
        feature_maps = F.relu(self.bn_last(self.conv_last(F.relu(self.bn4(feature_maps3_), inplace=True))),
                              inplace=True)
        feature_maps = feature_maps.reshape(B, -1, N, H, W)
        return feature_maps, feature_maps_d


# 双线性注意力池

class AttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, attentions, norm=2):
        H, W = features.size()[-2:]
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions = F.interpolate(attentions, size=(H, W), mode='bilinear', align_corners=True)
        if norm == 1:
            attentions = attentions+1e-8
        if len(features.shape) == 4:
            feature_matrix = torch.einsum('imjk,injk->imn', attentions, features)
        else:
            feature_matrix = torch.einsum('imjk,imnjk->imn', attentions, features)
        if norm == 1:
            w = torch.sum(attentions, dim=(2, 3)).unsqueeze(-1)
            feature_matrix /= w
        if norm == 2:
            feature_matrix = F.normalize(feature_matrix, p=2, dim=-1)
        if norm == 3:
            w = torch.sum(attentions, dim=(2, 3)).unsqueeze(-1)+1e-8
            feature_matrix /= w
        return feature_matrix


# 局部独立损失

class Auxiliary_Loss_v2(nn.Module):
    def __init__(self, M, N, C, alpha=0.05, margin=1, inner_margin=[0.1, 5]):
        super().__init__()
        self.register_buffer('feature_centers', torch.zeros(M, N))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.num_classes = C
        self.margin = margin
        self.atp = AttentionPooling()
        self.register_buffer('inner_margin', torch.Tensor(inner_margin))

    def forward(self, feature_map_d, attentions, y):
        B, N, H, W = feature_map_d.size()
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions = F.interpolate(attentions, (H, W), mode='bilinear', align_corners=True)
        feature_matrix = self.atp(feature_map_d,attentions)
        feature_centers = self.feature_centers
        center_momentum = feature_matrix-feature_centers
        real_mask = (y == 0).view(-1, 1, 1)
        fcts = self.alpha*torch.mean(center_momentum*real_mask,dim=0)+feature_centers
        fctsd = fcts.detach()
        if self.training:
            with torch.no_grad():
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(fctsd,torch.distributed.ReduceOp.SUM)
                    fctsd /= torch.distributed.get_world_size()
                self.feature_centers = fctsd
        inner_margin=self.inner_margin[y]
        intra_class_loss=F.relu(torch.norm(feature_matrix-fcts,dim=[1,2])*torch.sign(inner_margin)-inner_margin)
        intra_class_loss=torch.mean(intra_class_loss)
        inter_class_loss=0
        for j in range(M):
            for k in range(j+1, M):
                inter_class_loss += F.relu(self.margin-torch.dist(fcts[j],fcts[k]),inplace=False)
        inter_class_loss = inter_class_loss/M/self.alpha
        return intra_class_loss+inter_class_loss, feature_matrix


# 主干框架

class MARepVGG(nn.Module):
    def __init__(self, num_classes=2, M=8, mid_dims=256, alpha=0.05, margin=1, inner_margin=[0.01, 0.02]):
        super(MARepVGG, self).__init__()
        self.num_classes = num_classes
        self.M = M
        backbone = RepVGG(num_blocks=[2, 4, 14, 1],
                          num_classes=self.num_classes,
                          width_multiplier=[0.75, 0.75, 0.75, 2.5],
                          override_groups_map=None, deploy=False)
        self.layer0 = backbone.stage0
        self.layer1 = backbone.stage1
        self.layer2 = backbone.stage2
        self.layer3 = backbone.stage3
        self.layer4 = backbone.stage4
        self.attentions = AttentionMap(96, self.M)  # 利用注意力图定位伪造信息
        self.texture_enhance = Texture_Enhance_v2(48, 1)  # 在浅层提取纹理特征
        self.atp = AttentionPooling()  # 聚合texture和语义特征
        self.project_final = nn.Linear(1280, mid_dims)
        self.dropout_final = nn.Dropout(0.5, inplace=True)
        self.dropout = nn.Dropout2d(0.5, inplace=True)
        self.num_features = self.texture_enhance.output_features
        self.num_features_d = self.texture_enhance.output_features_d  # 1
        self.ensemble_classifier_fc = nn.Sequential(nn.Linear(mid_dims * 2, mid_dims), nn.Hardswish(),
                                                    nn.Linear(mid_dims, num_classes))
        self.projection_local = nn.Sequential(nn.Linear(8*self.num_features, mid_dims), nn.Hardswish(),
                                              nn.Linear(mid_dims, mid_dims))
        self.auxiliary_loss = Auxiliary_Loss_v2(M, self.num_features_d, num_classes, alpha, margin, inner_margin)  # 1

    def forward(self, x, y):  # 1
        x = self.layer0(x)
        for block in self.layer1:
            x = block(x)
        feature_maps, feature_maps_d = self.texture_enhance(x)  # 以layer1输出做输入提取纹理特征
        for block in self.layer2:
            x = block(x)
        attentions_map = self.attentions(x)  # 以layer2输出做输入生成多注意map
        attention_maps2 = attentions_map.sum(dim=1, keepdim=True)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)
        feature_matrix = self.atp(feature_maps_d, attentions_map)
        B, M, N = feature_matrix.size()
        aux_loss, feature_matrix_d = self.auxiliary_loss(feature_maps_d, attentions_map, y)  # 1
        feature_matrix = self.dropout(feature_matrix)
        feature_matrix = feature_matrix.view(B, -1)
        final = self.atp(x, attention_maps2, norm=1).squeeze(1)
        feature_matrix = F.hardswish(self.projection_local(feature_matrix))
        projected_final = F.hardswish(self.project_final(final))
        feature_matrix = torch.cat((feature_matrix, projected_final), 1)
        ensemble_logit = self.ensemble_classifier_fc(feature_matrix)
        ensemble_loss = F.cross_entropy(ensemble_logit, y)  # 1
        return dict(ensemble_logit=ensemble_logit, aux_loss=aux_loss, ensemble_loss=ensemble_loss)  # 1


if __name__ == '__main__':
    model = MARepVGG(num_classes=2, M=8)
    from thop import profile
    input = torch.rand(1, 3, 224, 224)
    y = torch.ones(1).type(torch.long)
    flops, params = profile(model, inputs=(input, y))
    total = sum([param.nelement() for param in model.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))
    model.eval()
    output = model(input, y)

