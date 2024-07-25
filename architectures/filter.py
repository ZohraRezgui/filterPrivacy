
import torch
from torch import nn


class FilterEstimator(nn.Module):
    def __init__(
        self,
        in_c,
        device,
        param_dict=None,
        init_lambda_zero=False,
    ):
        super(FilterEstimator, self).__init__()
        out_c = in_c
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=in_c // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(
            in_channels=in_c // 2, out_channels=out_c * 2, kernel_size=1
        )
        self.conv3 = nn.Conv2d(in_channels=out_c * 2, out_channels=out_c, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

        if init_lambda_zero:
            with torch.no_grad():
                nn.init.constant_(self.conv3.bias, -1.0)
                nn.init.xavier_normal_(self.conv3.weight)

        self.bound = 5
        self.to(device)

        self.device = device
        # self.index = index
        self.param_dict = param_dict


    def forward(self, R):
        m_r = torch.mean(R, dim=0) if self.param_dict is None else self.param_dict[0]
        std_r = torch.std(R, dim=0) if self.param_dict is None else self.param_dict[1]


        lambda_= self.conv1(R)
        lambda_ = self.relu(lambda_)
        lambda_ = self.conv2(lambda_)
        lambda_ = self.relu(lambda_)
        lambda_ = self.conv3(lambda_)


        lambda_ = lambda_.clamp(-self.bound, self.bound)
        lambda_ = self.sigmoid(lambda_)
        eps = torch.randn(size=R.shape).to(R.device) * std_r + m_r

        Z = R * lambda_ + (torch.ones_like(R).to(R.device) - lambda_) * eps




        return Z, lambda_



class IResNetFilter(torch.nn.Module):
    def __init__(self, backbone, iib):
        super(IResNetFilter, self).__init__()
        self.backbone = backbone
        self.iib = iib

    def forward(self, x):

        x0 = self.backbone.conv1(x)
        x = self.backbone.bn1(x0)
        x = self.backbone.prelu(x)
        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)
        x4, lambda_ = self.iib(x4) # restrict representation
        x = self.backbone.bn2(x4)
        x = torch.flatten(x, 1)
        x = self.backbone.dropout(x)
        x = self.backbone.fc(x)
        x = self.backbone.features(x)
        return x, lambda_

