import torch
import torchvision
import numpy as np
from models.vgg import Vgg16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

# Images will be normalized using these, because the CNNs were trained with normalized images as well!
IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)


model = Vgg16()
image = np.random.uniform(low=0, high=1, size=(3, 224, 224))
image = torch.tensor(image)
image = image.unsqueeze(0)
image = image.float()
image.requires_grad = True

LOWER_IMAGE_BOUND = torch.tensor((-IMAGENET_MEAN_1 / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to(DEVICE)
UPPER_IMAGE_BOUND = torch.tensor(((1 - IMAGENET_MEAN_1) / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to(DEVICE)

for iteration in range(1000):
    out = model(image)
    loss = torch.nn.MSELoss(reduction='mean')(out, torch.zeros_like(out))
    loss.backward()
    grad = image.grad.data
    g_std = torch.std(grad)
    g_mean = torch.mean(grad)
    grad = grad - g_mean
    grad = grad / g_std
    image.data += 0.05 * grad
    image.grad.data.zero_()
    image.data = torch.max(torch.min(image, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)

torchvision.transforms.ToPILImage()(image.detach().squeeze(0)).show()