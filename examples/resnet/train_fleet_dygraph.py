# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.vision.models import ResNet
from paddle.vision.models.resnet import BottleneckBlock
from paddle.io import Dataset, BatchSampler, DataLoader
import time

base_lr = 0.1
momentum_rate = 0.9
l2_decay = 1e-4

epoch = 10
batch_num = 100
batch_size = 32
class_dim = 102

# define a random dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        np.random.seed(fleet.worker_index()+1000)

    def __getitem__(self, idx):
        image = np.random.random([3, 224, 224]).astype('float32')
        label = np.random.randint(0, class_dim - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples

def optimizer_setting(parameter_list=None):
    optimizer = paddle.optimizer.Momentum(
        learning_rate=base_lr,
        momentum=momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(l2_decay),
        parameters=parameter_list)
    return optimizer


def train_resnet():
    fleet.init(is_collective=True)

    resnet = ResNet(BottleneckBlock, 50, num_classes=class_dim)
    optimizer = optimizer_setting(parameter_list=resnet.parameters())
    optimizer = fleet.distributed_optimizer(optimizer)
    resnet = fleet.distributed_model(resnet)

    dataset = RandomDataset(batch_num * batch_size)
    train_loader = DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=2)
    t1 = time.time()
    for eop in range(epoch):
        resnet.train()
        
        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True

            out = resnet(img)
            loss = paddle.nn.functional.cross_entropy(input=out, label=label)
            avg_loss = paddle.mean(x=loss)
            acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

            avg_loss.backward()
            optimizer.step()
            resnet.clear_gradients()

            if (batch_id+1) % 5 == 0:
                t2 = time.time()
                print("[Epoch %d, batch %d] loss: %.5f, acc1: %.5f, acc5: %.5f, speed: %.2f samples/s" % (eop, batch_id, avg_loss, acc_top1, acc_top5, (5*32/(t2 - t1))))
                t1 = time.time()

if __name__ == '__main__':
    train_resnet()
