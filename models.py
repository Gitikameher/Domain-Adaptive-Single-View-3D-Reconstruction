import torch
import torch.nn as nn
import torch.nn.functional as F

class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
       

    def forward(self, input):
        # input.shape == (bs,n,3)
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        xb = nn.Flatten(1)(xb)
        xb = F.relu(self.bn4(self.fc1(xb)))
        xb = F.relu(self.bn5(self.fc2(xb)))
      
      #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
        if xb.is_cuda:
            init=init.cuda()
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init
        return matrix


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.fc1 = nn.Conv1d(3,64,1)
        self.fc2 = nn.Conv1d(64,64,1) 
        self.fc4 = nn.Conv1d(64,128,1)
        self.fc5 = nn.Conv1d(128,1024,1)

        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, input):
        n_pts = input.size()[2]
        matrix3x3 = self.input_transform(input)
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.fc2(xb)))
        matrix128x128 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix128x128).transpose(1,2) 
        xb = F.relu(self.bn4(self.fc4(xb)))
        xb = self.bn5(self.fc5(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)        
        
        return xb

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128,2048*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
    def forward(self, input):
        input = input.view(input.size(0), -1)
        out = F.relu(self.bn1(self.fc1(input)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = F.relu(self.bn3(self.fc3(out)))
        out =  torch.tanh(self.fc4(out))
        return out.view(-1, 3, 2048)
        
ndf = 4
class _F_(nn.Module):
    def __init__(self):
        super(_F_, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(3, int(ndf/2) , 4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf/2) x 128 x 128
            nn.Conv2d(int(ndf/2), ndf, 4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16 
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, stride=1, padding=0, bias=False)
            # state size. 1
        )
    def forward(self, input):
        return torch.flatten(self.main(input), start_dim =1)

    
class Destructor_domain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128,1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
    def forward(self, input):
        input = input.view(input.size(0), -1)
        out = F.relu(self.bn1(self.fc1(input)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = F.relu(self.bn3(self.fc3(out)))
        return F.sigmoid(self.fc4(out))
    
    
class Destructor_shape(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128,1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
    def forward(self, input):
        input = input.view(input.size(0), -1)
        out = F.relu(self.bn1(self.fc1(input)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = F.relu(self.bn3(self.fc3(out)))
        return F.sigmoid(self.fc4(out))