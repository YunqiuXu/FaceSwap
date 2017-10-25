#!/usr/bin/env python

# Perform SR on many images
# sudo nice -n -19 ./super_resolution_batch.py

from __future__ import print_function
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
from glob import glob
import numpy as np

# paths
model_path = "model_scale_3_batch_4_epoch_500.pth"
input_path = "my_test/input_1/*.jpg"
# input_path = "my_test/input_2/*.png"
output_path = "my_test/output_1/"
# output_path = "my_test/output_2/"

# Load model
model = torch.load(model_path)

# Load images
for path in glob(input_path):
    img_name = path[-7:-4] # 001.jpg
    print("Processing " + img_name)

    img = Image.open(path).convert('YCbCr')
    y, cb, cr = img.split()
    input = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])

    out = model(input)
    out = out.cpu()

    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)

    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    out_img.save(output_path + img_name + ".png")

print("All finished")
