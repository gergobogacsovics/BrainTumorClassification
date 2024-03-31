import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import logging
import sys
import timm
import torchvision.models as models
from histogramloss import L2Normalization

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = True

class BaseNetwork(Enum):
    ALEXNET = "AlexNet"
    MOBILENETV2 = "MobileNet V2"
    EFFICIENTNET_B0 = "EfficientNetB0"
    SHUFFLENET_V2_1_0= "ShuffleNetV2_1_0"
    VGG16 = "VGG16"

def load_weights(from_, to):
    model_dict = to.state_dict()
    from_dict = from_.state_dict()

    pretrained_dict = {k: v for k, v in from_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)

    to.load_state_dict(model_dict)

    return to

def get_vgg16(pretrained):
    model = models.vgg16(pretrained=pretrained)
    features = model.features
    classifier = model.classifier

    return features, classifier

def get_mobilenet_v2(pretrained):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=pretrained)

    features = model.features

    classifier = model.classifier

    return features, classifier

def get_efficientnet_b0(pretrained):
    model = models.efficientnet_b0(pretrained=pretrained)
    features = model.features
    classifier = model.classifier

    return features, classifier

def get_shufflenet_v2_1_0(pretrained):
    features = models.shufflenet_v2_x1_0(pretrained=pretrained)
    classifier = features.fc
    features.fc = nn.Identity()
    
    return features, classifier

def get_base_network(name, num_channels, img_size, num_classes, pretrained, keep_original_architecture, embedding_size=None):
    if name == BaseNetwork.ALEXNET.value:
        model_alex = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=pretrained)
        cnn_out_shape = model_alex.features(torch.zeros(1, 3, img_size, img_size)).view(-1).shape[0]
        logging.info(f"Feed forward first layer will have {cnn_out_shape} neurons.")

        features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=9216, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        )

        if not keep_original_architecture:
            logging.info("Using MODIFIED ensemble architecture.")
            embedding_layer = nn.Linear(in_features=cnn_out_shape, out_features=embedding_size)
            classifier[1] = nn.Linear(in_features=embedding_size, out_features=4096, bias=True)
        else:
            logging.info("Using ORIGINAL architecture.")
            embedding_layer = None

        print("classifier", classifier)

    elif name == BaseNetwork.MOBILENETV2.value:
        features, classifier = get_mobilenet_v2(pretrained=pretrained)
        features.add_module("avg_pool_fix", nn.AdaptiveAvgPool2d((1,1)))

        if num_channels != 3:
            features[0][0] = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            logging.info(f"Setting number of channels to {num_channels}.")

        dummy_img = torch.zeros(1, num_channels, img_size, img_size)

        cnn_out_shape = features(dummy_img).view(-1).shape[0]
        logging.info(f"Feed forward first layer will have {cnn_out_shape} neurons.")

        if not keep_original_architecture:
            logging.info("Using MODIFIED ensemble architecture.")
            embedding_layer = nn.Linear(in_features=cnn_out_shape, out_features=embedding_size)
            classifier[1] = nn.Linear(in_features=embedding_size, out_features=num_classes)
        else:
            logging.info("Using ORIGINAL architecture.")
            embedding_layer = None
            classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
        
        print("classifier", classifier)
    elif name == BaseNetwork.EFFICIENTNET_B0.value:
        features, classifier = get_efficientnet_b0(pretrained=pretrained)
        features.add_module("avg_pool_fix", nn.AdaptiveAvgPool2d(output_size=1))

        if num_channels != 3:
            features[0][0] = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            logging.info(f"Setting number of channels to {num_channels}.")
        
        cnn_out_shape = features(torch.zeros(1, num_channels, img_size, img_size)).view(-1).shape[0]
        logging.info(f"Feed forward first layer will have {cnn_out_shape} neurons.")

        if not keep_original_architecture:
            logging.info("Using MODIFIED ensemble architecture.")
            embedding_layer = nn.Linear(in_features=cnn_out_shape, out_features=embedding_size)
            classifier[-1] = nn.Linear(in_features=embedding_size, out_features=num_classes, bias=True)
        else:
            logging.info("Using ORIGINAL architecture.")
            embedding_layer = None
            classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        
        print("classifier", classifier)
    elif name == BaseNetwork.SHUFFLENET_V2_1_0.value:
        features, classifier = get_shufflenet_v2_1_0(pretrained=pretrained)

        if num_channels != 3:
            features.conv1[0] = nn.Conv2d(num_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            logging.info(f"Setting number of channels to {num_channels}.")
        
        cnn_out_shape = features(torch.zeros(1, num_channels, img_size, img_size)).view(-1).shape[0]
        logging.info(f"Feed forward first layer will have {cnn_out_shape} neurons.")

        if not keep_original_architecture:
            logging.info("Using MODIFIED ensemble architecture.")
            embedding_layer = nn.Linear(in_features=cnn_out_shape, out_features=embedding_size)
            classifier = nn.LinearLinear(in_features=embedding_size, out_features=num_classes, bias=True)
        else:
            logging.info("Using ORIGINAL architecture.")
            embedding_layer = None
            classifier = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        
        print("classifier", classifier)
    elif name == BaseNetwork.VGG16.value:
        logging.info("Using VGG16 network")
        features, classifier = get_vgg16(pretrained=pretrained)
        features.add_module("avg_pool_fix", nn.AdaptiveAvgPool2d(output_size=(7, 7)) )
        
        logging.info(f"NUM CHANNELS: {num_channels}")

        if num_channels != 3:
            features[0] = nn.Conv2d(num_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            logging.info(f"Setting number of channels to {num_channels}.")

        cnn_out_shape = features(torch.zeros(1, num_channels, img_size, img_size)).view(-1).shape[0]
        logging.info(f"Feed forward first layer will have {cnn_out_shape} neurons.")
        
        if not keep_original_architecture:
            logging.info("MODIFIED ensemble architecture is DISABLED for VGG16.")
            exit(-1)
        else:
            logging.info("Using ORIGINAL architecture.")
            embedding_layer = None
            classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
    else:
        logging.error(f"Unknown base network name '{name}'.")
        sys.exit(-1)

    return features, classifier, embedding_layer


def get_base_network_output_dim(name, img_size=None):
    if name == BaseNetwork.ALEXNET.value:
        return 9216
    elif name == BaseNetwork.SHUFFLENET_V2_1_0.value:
        return 1024
    elif name == BaseNetwork.EFFICIENTNET_B0.value:
        return 1280
    elif name == BaseNetwork.MOBILENETV2.value:
        return 1280
    elif name == BaseNetwork.VGG16.value:
        return 25088
    else:
        logging.error(f"Unknown base network name '{name}'.")
        sys.exit(-1)

class BaseNetworkWrapperOriginalNOFLATTEN(nn.Module):
    def __init__(self, base_network, num_channels, num_classes, img_size, pretrained):
        super().__init__()
        
        self.features, self.classifier, _ = get_base_network(base_network, num_channels=num_channels, img_size=img_size,
            num_classes=num_classes, pretrained=pretrained, embedding_size=None, keep_original_architecture=True)
    def forward(self, images):
        feats = self.features(images)
        
        feats_flattened = torch.flatten(feats, 1)

        outs = self.classifier(feats)

        return feats_flattened, outs

class BaseNetworkWrapperOriginal(nn.Module):
    def __init__(self, base_network, num_channels, num_classes, img_size, pretrained):
        super().__init__()
        
        self.features, self.classifier, _ = get_base_network(base_network, num_channels=num_channels, img_size=img_size,
            num_classes=num_classes, pretrained=pretrained, embedding_size=None, keep_original_architecture=True)
    def forward(self, images):
        feats = self.features(images)
        
        feats = torch.flatten(feats, 1)

        outs = self.classifier(feats)

        return feats, outs
    
class BaseNetworkWrapperL2Normed(nn.Module):
    def __init__(self, base_network, num_channels, num_classes, img_size, pretrained):
        super().__init__()
        
        self.features, self.classifier, _ = get_base_network(base_network, num_channels=num_channels, img_size=img_size,
            num_classes=num_classes, pretrained=pretrained, embedding_size=None, keep_original_architecture=True)
        
        self.l2_norm = L2Normalization()

    def forward(self, images):
        feats = self.features(images)
        
        feats = torch.flatten(feats, 1)

        outs = self.classifier(feats)

        return self.l2_norm(feats), outs

class BaseNetworkWrapper(nn.Module):
    def __init__(self, base_network, num_channels, num_classes, img_size, pretrained, embedding_size):
        super().__init__()
        
        self.features, self.classifier, self.embedding_layer = get_base_network(base_network, num_channels=num_channels, img_size=img_size,
            num_classes=num_classes, pretrained=pretrained, embedding_size=embedding_size, keep_original_architecture=False)

    def forward(self, images):
        feats = self.features(images)

        feats = torch.flatten(feats, 1)

        feats = self.embedding_layer(feats)

        outs = self.classifier(feats)

        return feats, outs


class CNNEnsemble_ConvFeaturesOnly(nn.Module):
    def __init__(self, base_network_1, base_network_2, base_network_3):
        super().__init__()
        
        self.base_network_1 = base_network_1
        self.base_network_2 = base_network_2
        self.base_network_3 = base_network_3
        
    def forward(self, images):
        feats_1, outs_1 = self.base_network_1(images)
        feats_2, outs_2 = self.base_network_2(images)
        feats_3, outs_3 = self.base_network_3(images)

        return feats_1, outs_1, feats_2, outs_2, feats_3, outs_3