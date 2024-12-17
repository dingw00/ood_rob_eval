import torch
import torch.nn as nn
from torchvision.models.inception import Inception3 as TV_Inception3, Inception_V3_Weights

class Inception3(TV_Inception3):
    def __init__(self, num_classes=50):
        super(Inception3, self).__init__(aux_logits=True)

        self.load_state_dict(Inception_V3_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True))

        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, num_classes)

        self.layer_list = ["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "maxpool1", "Conv2d_3b_1x1", "Conv2d_4a_3x3",
                           "maxpool2", "Mixed_5b", "Mixed_5c", "Mixed_5d", "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d",
                           "Mixed_6e", "Mixed_7a", "Mixed_7b", "Mixed_7c", "avgpool", "fc", "fc2"]
        
    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        x = self.dropout2(x)
        x = self.fc2(x)
        # N x 50 (num_classes)
        return x

    def forward_features(self, x, layer):
        assert layer in self.layer_list

        x = self.Conv2d_1a_3x3(x)
        if layer == self.layer_list[0]:
            return x
        x = self.Conv2d_2a_3x3(x)
        if layer == self.layer_list[1]:
            return x
        x = self.Conv2d_2b_3x3(x)
        if layer == self.layer_list[2]:
            return x
        x = self.maxpool1(x)
        if layer == self.layer_list[3]:
            return x
        x = self.Conv2d_3b_1x1(x)
        if layer == self.layer_list[4]:
            return x
        x = self.Conv2d_4a_3x3(x)
        if layer == self.layer_list[5]:
            return x
        x = self.maxpool2(x)
        if layer == self.layer_list[6]:
            return x
        x = self.Mixed_5b(x)
        if layer == self.layer_list[7]:
            return x
        x = self.Mixed_5c(x)
        if layer == self.layer_list[8]:
            return x
        x = self.Mixed_5d(x)
        if layer == self.layer_list[9]:
            return x
        x = self.Mixed_6a(x)
        if layer == self.layer_list[10]:
            return x
        x = self.Mixed_6b(x)
        if layer == self.layer_list[11]:
            return x
        x = self.Mixed_6c(x)
        if layer == self.layer_list[12]:
            return x
        x = self.Mixed_6d(x)
        if layer == self.layer_list[13]:
            return x
        x = self.Mixed_6e(x)
        if layer == self.layer_list[14]:
            return x
        x = self.Mixed_7a(x)
        if layer == self.layer_list[15]:
            return x
        x = self.Mixed_7b(x)
        if layer == self.layer_list[16]:
            return x
        x = self.Mixed_7c(x)
        if layer == self.layer_list[17]:
            return x
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        if layer == self.layer_list[18]:
            return x
        x = self.fc(x)
        x = self.dropout2(x)
        if layer == self.layer_list[19]:
            return x
        x = self.fc2(x)
        if layer == self.layer_list[20]:
            return x
    
    def forward_head(self, x, layer):
        assert layer in self.layer_list
        
        if layer in self.layer_list[:1]:
            x = self.Conv2d_2a_3x3(x)
        if layer in self.layer_list[:2]:
            x = self.Conv2d_2b_3x3(x)
        if layer in self.layer_list[:3]:
            x = self.maxpool1(x)
        if layer in self.layer_list[:4]:
            x = self.Conv2d_3b_1x1(x)
        if layer in self.layer_list[:5]:
            x = self.Conv2d_4a_3x3(x)
        if layer in self.layer_list[:6]:
            x = self.maxpool2(x)
        if layer in self.layer_list[:7]:
            x = self.Mixed_5b(x)
        if layer in self.layer_list[:8]:
            x = self.Mixed_5c(x)
        if layer in self.layer_list[:9]:
            x = self.Mixed_5d(x)
        if layer in self.layer_list[:10]:
            x = self.Mixed_6a(x)
        if layer in self.layer_list[:11]:
            x = self.Mixed_6b(x)
        if layer in self.layer_list[:12]:
            x = self.Mixed_6c(x)
        if layer in self.layer_list[:13]:
            x = self.Mixed_6d(x)
        if layer in self.layer_list[:14]:
            x = self.Mixed_6e(x)
        if layer in self.layer_list[:15]:
            x = self.Mixed_7a(x)
        if layer in self.layer_list[:15]:
            x = self.Mixed_7b(x)
        if layer in self.layer_list[:17]:
            x = self.Mixed_7c(x)
        if layer in self.layer_list[:18]:
            x = self.avgpool(x)
            x = self.dropout(x)
            x = torch.flatten(x, 1)
        if layer in self.layer_list[:19]:
            x = self.fc(x)
            x = self.dropout2(x)
        if layer in self.layer_list[:20]:
            x = self.fc2(x)
        return x
    
