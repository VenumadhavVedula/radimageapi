import os,sys
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import skimage, skimage.io

import torch
import torchvision

import torchxrayvision as xrv

def heatmap_xray(image_path, heatmap_path=None):
    img = skimage.io.imread(image_path)
    img = xrv.datasets.normalize(img, 255)  

    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")

    # Add color channel
    img = img[None, :, :]

    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224)])

    img = transform(img)
    img = torch.from_numpy(img).unsqueeze(0)
    model = xrv.models.get_model('densenet121-res224-all')
    target = model.pathologies.index('Mass')

    img = img.requires_grad_()

    outputs = model(img)
    print(outputs[:,target])
    grads = torch.autograd.grad(outputs[:,target], img)[0][0][0]
    blurred = skimage.filters.gaussian(grads.detach().cpu().numpy()**2, sigma=(5, 5), truncate=3.5)

    my_dpi = 100
    fig = plt.figure(frameon=False, figsize=(224/my_dpi, 224/my_dpi), dpi=my_dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img[0][0].detach().cpu().numpy(), cmap="gray", aspect='auto')
    ax.imshow(blurred, alpha=0.5)
    plt.savefig(heatmap_path, dpi=my_dpi, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)