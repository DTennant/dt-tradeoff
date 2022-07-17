from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet50, ResNet18, ResNet34, ResNet101,ResNet152
from .wrn import wrn_16_1, wrn_16_2, wrn_16_4, wrn_40_1, wrn_40_2, wrn_40_4, wrn_28_2, wrn_28_4, wrn_64_4, wrn_28_10, wrn_16_10, wrn_22_4, wrn_34_4, wrn_46_4, wrn_52_4
from .mobilenetv2 import mobile_half
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .resnext import resnext32_4x16d, resnext32_8x8d, resnext32_16x4d

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'ResNet152': ResNet152,
    'ResNet101': ResNet101,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_16_4': wrn_16_4,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'wrn_40_4': wrn_40_4,
    'wrn_28_2': wrn_28_2,
    'wrn_28_4': wrn_28_4,
    'wrn_34_4': wrn_34_4,
    'wrn_22_4': wrn_22_4,
    'wrn_46_4': wrn_46_4,
    'wrn_52_4': wrn_52_4,
    'wrn_64_4': wrn_64_4,
    'wrn_28_10': wrn_28_10,
    'wrn_16_10': wrn_16_10,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'resnext32_8x8d': resnext32_8x8d,
    'resnext32_4x16d': resnext32_4x16d,
    'resnext32_16x4d': resnext32_16x4d,
}
