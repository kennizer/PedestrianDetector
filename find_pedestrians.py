
import argparse
import glob
import random
import time
import numpy as np
import torch
import sys
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from PIL import Image, ImageDraw
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3360, 40)
        self.fc2 = nn.Linear(40, 2)
        self.conv1 = nn.Conv2d(3,10,(5,5))
        self.maxpool = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(10,20,(3,3))
        self.conv3 = nn.Conv2d(20,40,(3,3))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x= self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x= self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x= x.view(-1, 3360)
        x = self.fc1(x)
        x= self.fc2(x)
        return x
class dataset(object):
    def __init__(self, path):
        # neg is 0, pos is 1
        pos_im = glob.glob(path + '/pos/*.png')
        neg_im = glob.glob(path + '/neg/*.png')
        img = [(x, 1) for x in pos_im]
        img = img + [(x, 0) for x in neg_im]
        random.shuffle(img)
        self.data = img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index][0]).resize((64, 128))
        img = np.array(img).transpose((2, 0, 1))[:3]
        img = img / 255. - 0.5
        img = torch.from_numpy(img).float()
        label = self.data[index][1]
        return img, label


def train(model, loader, optimizer, criterion, epoch, device):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (input, target) in enumerate(loader):
        input = Variable(input)
        target = Variable(target)
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        accuracy = (output.topk(1)[1].transpose(0,1) == target).float().mean()
        loss.backward()
        optimizer.step()
        losses.update(loss.item())
        accuracies.update(accuracy)
    print('Train: epoch {}\t loss {}\t accuracy {}'.format(epoch, losses.avg, accuracies.avg))

def test(model, loader, criterion, epoch, device):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (input, target) in enumerate(loader):
        input = Variable(input)
        target = Variable(target)
        input, target = input.to(device), target.to(device)
        output = model(input)
        loss = criterion(output, target)
        accuracy = (output.topk(1)[1].transpose(0,1) == target).float().mean()
        losses.update(loss.item())
        accuracies.update(accuracy)
    print('Test: epoch {}\t loss {}\t accuracy {}'.format(epoch, losses.avg, accuracies.avg))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def multi_scale_scan(model,img):
    lst = [] 
    x_size,y_size = img.size 
    scale = 1
    while x_size>=64 and y_size>=128:
        t_list = scan_image(model,img)
        for (x,y,score) in t_list:
            item = (int(round(x/scale)),int(round(y/scale)),int(64/scale),int(128/scale),score)
            lst.append(item)
            pass 
        x_size, y_size = img.size 
        img = img.resize((int(round(x_size*0.8)),(int(round(y_size*0.8)))))
        scale*=0.8
    return lst
def scan_image(model,img):
    lst = []
    for i in range(0, img.size[0], 16):
        for j in range (0, img.size[1], 16):
            imgc = img.crop((i,j, i+64, j+128))
            dec = query(model,imgc)
            if dec>9: 
                lst.append((i,j,dec))
    return lst
def query (model, img):
    assert img.size[0]==64 and img.size[1]==128 
    img = np.array(img).transpose((2, 0, 1))[:3]
    img = img / 255. - 0.5
    img = torch.from_numpy(img).float()
    img = torch.unsqueeze(img,0)
    res = model(img).tolist()
    return res[0][1] - res[0][0] 
if __name__ == '__main__':
    model = Net() 
    model.load_state_dict(torch.load('model.pth.tar'))
    model.eval() 
    image_location = sys.argv[1]
    output = sys.argv[2]
    im = Image.open(image_location)
    lst = scan_image(model,im)
    draw = ImageDraw.Draw(im)
    lst = multi_scale_scan(model,im)
    boxes = []
    scores = [] 
    for (x,y, width, height, score) in lst: 
        boxes.append((float(x),float(y),float(x+width),float(y+height)))
        scores.append(float(score))
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    selected = torchvision.ops.nms(boxes, scores, 0.25).tolist() 
    for item in selected:
        (x,y, width,height, score) = lst[item] 
        draw.rectangle((x,y, x+width, y+height), outline=(0, 255, 0))
    im.save(output)