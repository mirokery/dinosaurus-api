import torch.nn as nn
import torch.nn.functional as F

class MyArchitecture(nn.Module):
  def __init__(self,num_channels,num_classes ):
    super(MyArchitecture,self).__init__()
    self.conv1 = nn.Conv2d(num_channels,64,(3,3))
    self.conv2 = nn.Conv2d(64,64,(3,3))
    self.conv3 = nn.Conv2d(64,128,(3,3))
    self.conv4 = nn.Conv2d(128,256,(3,3))
    self.lin1 =nn.Linear(5161984,50)
    self.lin2 =nn.Linear(50,num_classes)

  def forward(self,x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = x.reshape(x.shape[0],-1)
    x = F.relu(self.lin1(x))
    x = self.lin2(x)

    return x
