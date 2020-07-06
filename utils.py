import torch
import os
import numpy as np
import random
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def preprocess(image, device):
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), SubtractMean()])
    image = transform(image)

    return image.to(device)

# Transform img


class SubtractMean(object):
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])

    def __call__(self, img):
        img = np.array(img, dtype=np.uint8)
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img


def euclidean_distance(feature1, feature2):
    print(f'{feature1.device}{feature2.device}')
    return torch.cdist(feature1.view(1, -1), feature2.view(1, -1))
