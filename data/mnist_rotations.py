# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision import transforms
from PIL import Image
import random
import torch


def rotate_dataset(d, rotation):
    result = torch.FloatTensor(d.size(0), 784)
    tensor = transforms.ToTensor()

    for i in range(d.size(0)):
        img = Image.fromarray(d[i].numpy(), mode='L')
        result[i] = tensor(img.rotate(rotation)).view(784)
    return result


output_file = 'mnist_rotations.pt'
tasks_num = 20
min_rotation = 0
max_rotation = 180
random_seed = 0

torch.manual_seed(random_seed)

tasks_tr = []
tasks_te = []

x_tr, y_tr = torch.load('mnist_train.pt')
x_te, y_te = torch.load('mnist_test.pt')

for t in range(tasks_num):
    min_rot = 1.0 * t / tasks_num * (max_rotation - min_rotation) + \
              min_rotation
    max_rot = 1.0 * (t + 1) / tasks_num * \
              (max_rotation - min_rotation) + min_rotation
    rot = random.random() * (max_rot - min_rot) + min_rot

    tasks_tr.append([rot, rotate_dataset(x_tr, rot), y_tr])
    tasks_te.append([rot, rotate_dataset(x_te, rot), y_te])

torch.save([tasks_tr, tasks_te], output_file)
