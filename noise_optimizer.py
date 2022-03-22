import torch
import numpy as np
from torch.optim import Adam
from torchvision import models
from PIL import Image
import copy

model = models.vgg16(pretrained=True).eval()

image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
image = Image.fromarray(image)

def process(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    im_as_arr = np.float32(image)
    im_as_arr = im_as_arr.transpose(2, 0, 1)
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]

    im_as_ten = torch.from_numpy(im_as_arr).float()
    im_as_ten = im_as_ten.unsqueeze(0)
    im_as_ten.requires_grad = True
    return im_as_ten

def recreate_image(im_as_var):
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im

processed_image = process(image)
optimizer = Adam([processed_image], lr=0.1)
for _ in range(150):
    output = model(processed_image)
    class_loss = -output.squeeze(0)[130] # Class (current: 100)
    processed_image.grad = None
    class_loss.backward()
    optimizer.step()

image = recreate_image(processed_image)

Image.fromarray(image).save("/Users/arnavshah/Downloads/optimized_noise.jpeg") # Replace with image path