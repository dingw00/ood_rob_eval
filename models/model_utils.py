import torch

# import models.resnetv2 as resnetv2
import models.cifar10.wrn as wrn
from torchvision.models.resnet import Bottleneck as Bottleneck_Imagenet
from .cifar10.resnet import ResNet, BasicBlock, Bottleneck, ResNet_Imagenet
from .awa2.inception import Inception3
import torchvision as tv
from torchvision.models.vision_transformer import ViT_B_16_Weights
import timm
import os

def load_model(model_name, weight_path=None, benchmark="CIFAR10", device="cpu"):

    if benchmark == "Imagenet1k":
        num_classes = 1000
        if model_name == "swin" and (weight_path is not None):
            model = timm.create_model(os.path.split(weight_path)[-1], pretrained=True)
        elif model_name == "deit" and (weight_path is not None):
            model = timm.create_model(os.path.split(weight_path)[-1], pretrained=True)
        elif model_name == "vit":
            if "vit_b_16" in weight_path: 
                model = tv.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        if benchmark == "CIFAR10":
            num_classes = 10
            if model_name == 'wrn_40_2':
                model = wrn.KNOWN_MODELS[model_name.upper().replace("_","-")](head_size=num_classes, dropRate=0.3)
            elif model_name == "resnet50":
                # model = resnet_cifar.resnet50(num_classes=num_classes)
                model = ResNet(Bottleneck, num_classes=num_classes)
            elif model_name == "resnet":
                model = ResNet(BasicBlock, num_classes=num_classes)
        
        elif benchmark == "Imagenet100":
            num_classes = 100
            if model_name == "resnet50":
                model = ResNet_Imagenet(Bottleneck_Imagenet, num_classes=num_classes)

        elif benchmark == "AwA2":
            num_classes = 50
            if model_name == "inception_v3":
                model = Inception3(num_classes=num_classes)

        if weight_path:
            state_dict = torch.load(weight_path, map_location=device)
            if "model" in state_dict:
                model.load_state_dict(state_dict["model"])
            else:
                model.load_state_dict(state_dict)
        
    model.to(device)
    return model

class InputNormalizer(torch.nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    '''
    def __init__(self, benchmark=None, model_arch=None, model_weight=None, mean=[1, 1, 1], std=[0, 0, 0]):
        super(InputNormalizer, self).__init__()

        if benchmark == "CIFAR10":

            # for PAT and robustness repo
            if model_arch == "wrn_40_2":
                
                if "Hendrycks2020AugMix" in model_weight:
                    # for AT - wrn_40_2_Hendrycks2020AugMix.pt
                    mean = [0.5, 0.5, 0.5]
                    std = [0.5, 0.5, 0.5]
                else:
                    mean = [0.49139968, 0.48215827, 0.44653124]
                    std = [0.24703233, 0.24348505, 0.26158768]

            elif model_arch == "resnet50":
                # for PAT and robustness repo
                mean = [0.4914, 0.4822, 0.4465]
                std = [0.2023, 0.1994, 0.2010]

        elif benchmark == "Imagenet100":

            if model_arch == "resnet50":
                mean = [0.4717, 0.4499, 0.3837]
                std = [0.2600, 0.2516, 0.2575]
        
        elif benchmark == "Imagenet1k":
            if model_arch in ["swin", "deit", "vim"]:
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]


        mean = torch.tensor(mean)
        std = torch.tensor(std)
        self.mean = mean[None, ..., None, None]
        self.std = std[None, ..., None, None]
                                     
    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        if x.get_device() == -1:
            x_normalized = (x - self.mean)/self.std
        else:
            x_normalized = (x - self.mean.cuda())/self.std.cuda()
        return x_normalized