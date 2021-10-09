import torch.nn as nn
import torch
import torch.nn.functional as F

device = 'cuda:0'

class CNN_LSTM_RIGHT(nn.Module):
    def __init__(self):
        super(CNN_LSTM_RIGHT, self).__init__()

        #self.weights_init_normal()
        
        self.relu = F.leaky_relu
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=0, bias=True)
        self.batnorm1 = nn.BatchNorm2d(16)
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=6, stride=1, padding=0, bias=True)
        self.batnorm2 = nn.BatchNorm2d(32)
        self.conv3_1 = nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=0, bias=True)
        self.batnorm3 = nn.BatchNorm2d(48)
        self.conv4_1 = nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=0, bias=True)
        self.batnorm4 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(6400, 1024)
        self.batnorm5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.batnorm6 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 128)
        self.batnorm7 = nn.BatchNorm1d(128)
        
        self.lstm = nn.LSTM(10,512,3)
        
        self.fc4 = nn.Linear(3, 32)
        self.batnorm8 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(81920,60)
        self.batnorm9 = nn.BatchNorm1d(60)
        
        #self.fc5 = nn.Linear(1600,60)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
    # weight initialization
    def weights_init_normal(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model
        if classname.find('Linear') != -1:
            y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
            m.weight.data.normal_(0.0,1/np.sqrt(y))
        # m.bias.data should be 0
            m.bias.data.fill_(0)

    # 사진 시퀀스와 좌표,속력 시퀀스를 받음
    def forward(self, x, pos):
        bat, seq = x.shape[:2]
        x = x.view(-1, 3, 224, 224)
        x = self.relu(self.conv1_1(x))
        x = self.maxpool(x)
        x = self.batnorm1(x)
        x = self.relu(self.conv2_1(x))
        x = self.maxpool(x)
        x = self.batnorm2(x)
        x = self.relu(self.conv3_1(x))
        x = self.maxpool(x)
        x = self.batnorm3(x)
        x = self.relu(self.conv4_1(x))
        x = self.maxpool(x)
        x = self.batnorm4(x)
        
        x = x.view(bat,seq,-1)
        x = self.fc1(x)
        x = x.view(bat,x.shape[2],seq)
        x = self.batnorm5(x)
        x = x.view(bat,seq,-1)
        x = self.fc2(x)
        x = x.view(bat,x.shape[2],seq)
        x = self.batnorm6(x)
        x = x.view(bat,seq,-1)
        x = self.fc3(x)
        x = x.view(bat,x.shape[2],seq)
        x = self.batnorm7(x)
        
        pos = self.fc4(pos)
        pos = pos.view(bat, pos.shape[2], seq)
        pos = self.batnorm8(pos)
        
        cat = torch.cat((x,pos),1)
        
        out, _ = self.lstm(cat)
        out = out.view(bat,-1)
        
        y = self.fc5(out)
        y = torch.reshape(y, (bat,20,3))
                
        return y


