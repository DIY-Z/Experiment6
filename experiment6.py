import numpy as np
import torch
from utils import get_dataloader
from models import AutoEncoder, StackedAutoEncoder
from torch.nn import Linear, CrossEntropyLoss


def train_annual_layer(layers, idx, max_epoch = 20):
    """
    idx:指当前训练哪一层,再训练该层时会冻结当前层前面的层的参数
    """
    if torch.cuda.is_available():
        for model in layers:
            model.cuda()
    train_dataloader, _ = get_dataloader()
    optimizer = torch.optim.SGD(layers[idx].parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    epoch = 0
    while epoch < max_epoch:
        
        for i in range(idx):
            for param in layers[i].parameters():
                param.requires_grad = False
            layers[i].flag = False

        for batch_idx, (train_data, _) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                train_data = train_data.cuda()
            out = train_data.view(train_data.size(0), -1)
            for l in range(idx):
                out = layers[l](out)    
            pred = layers[idx](out)

            optimizer.zero_grad()
            loss = criterion(pred, out) #注意:pred和out不能互换
            loss.backward()
            optimizer.step()
            # if (batch_idx + 1) % 10 == 0:
            #     print(f'Loss{loss}')
        epoch += 1

def train_whole(model, max_epoch = 50):
    print("---------train_whole-----------")
    if torch.cuda.is_available():
        model.cuda()
    for param in model.parameters():
        param.require_grad = True
    
    train_dataloader, _ = get_dataloader()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    epoch = 0
    while epoch < max_epoch:
        for batch_idx, (train_data, _) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                train_data = train_data.cuda()
            out = train_data.view(train_data.size(0), -1)
            pred = model(out)
            
            optimizer.zero_grad()
            loss = criterion(pred, out)
            loss.backward()
            optimizer.step()
            # if (batch_idx + 1) % 10 == 0:
            #     print(f'Loss{loss}')

        epoch += 1

def train_classifier(model, max_epoch = 50):
    if torch.cuda.is_available():
        model.cuda()
    train_dataloader, test_dataloader = get_dataloader()

    classifier = Linear(64, 10).cuda()
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    epoch = 0
    while epoch < max_epoch:
        correct = 0
        for batch_idx, (train_data, train_label) in enumerate(train_dataloader):
            target = train_label.cuda()
            img = train_data.cuda()
            features = model(img).detach()
            prediction = classifier(features.view(features.size(0), -1))
            loss = criterion(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred = prediction.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        print('classifier accuracy : {}/{}={:.2f}%'.format(correct, len(train_dataloader)*16, 100*float(correct)/(len(train_dataloader)*16)))
        epoch += 1
if __name__ == '__main__':
    encoder1 = AutoEncoder(784, 256, flag=True)
    encoder2 = AutoEncoder(256, 64, flag=True)
    encoder3 = AutoEncoder(64, 256, flag=True)
    encoder4 = AutoEncoder(256, 784, flag=True)
    layers = [encoder1, encoder2, encoder3, encoder4]

    #按照顺序对每一层单独进行预训练
    for idx in range(4):
        train_annual_layer(layers=layers,idx=idx)

    stkAuto = StackedAutoEncoder(layers=layers)

    #对所有层都一起训练
    train_whole(model=stkAuto)

    stkAuto.flagx = True
    #微调,训练分类器
    train_classifier(model=stkAuto)