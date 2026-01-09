import torch
import torch.nn as nn
from torchvision.models import resnet,vgg
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self, dim, hdim, r=8):
        super(Attention, self).__init__()
        self.dim = dim
        self.hdim = hdim
        self.r = r
        self.layer2 = nn.Sequential(
            nn.Linear(self.dim, self.r),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(self.r, self.hdim),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.layer2(input)
        x = self.layer3(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return x


class Local(nn.Module):
    def __init__(self, dim, hdim, r=16,num_att=6):
        super(Local, self).__init__()
        self.dim = dim
        self.hdim = hdim
        self.r = r
        self.num_att=num_att
        self.layer = Attention(dim, hdim, self.r)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, p):
        a = self.avgpool(p).squeeze()
        a = self.layer(a)
        a = a.view(a.shape[0], a.shape[1], 1, 1)
        a = a * p
        a = a.sum(1).view(a.shape[0], -1)
        a = torch.softmax(a, dim=1)
        a = a.view(a.shape[0], 1, p.shape[2], p.shape[3])
        return self.avgpool(a * p).squeeze()

class Fusion(nn.Module):
    def __init__(self, dim=512, hdim=512, r=8,num_att=6):
        super(Fusion, self).__init__()
        self.dim = dim
        self.hdim = hdim
        self.r = r
        self.num_att=num_att
        self.dropout=nn.Dropout()
        self.layer2 = nn.Sequential(
            nn.Linear(self.dim, self.r),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(self.r, self.hdim),
            nn.ReLU()
        )

    def forward(self,x1,x2,aa=False):
        if aa==True:
            x2=self.dropout(x2)
        x1_=x1.view(-1,self.num_att,self.dim).view(-1,self.dim)
        x2_=x2.view(-1,self.num_att,self.dim).view(-1,self.dim)
        x1_=self.layer2(x1_)
        x1_=self.layer3(x1_)
        x2_=self.layer2(x2_)
        x2_=self.layer3(x2_)
        x1_=x1_.view(-1,self.num_att,self.dim)
        x2_=x2_.view(-1,self.num_att,self.dim)
        Fx1_=F.normalize(x1_,dim=2)
        Fx2_=F.normalize(x2_,dim=2)
        S=Fx1_.matmul(Fx2_.permute(0,2,1))
        return x1+S.matmul(x2.view(-1,self.num_att,self.dim)).view(-1,self.num_att*self.dim)


class Model(nn.Module):
    def __init__(self, num_classes=1000,num_att=2):
        super(Model, self).__init__()
        self.num_classes=num_classes
        self.num_att=num_att
        ResNet=resnet.resnet18(pretrained=True)
        self.conv1=ResNet.conv1
        self.bn1=ResNet.bn1
        self.relu=ResNet.relu
        self.maxpool=ResNet.maxpool
        self.layer1=ResNet.layer1
        self.layer2=ResNet.layer2
        self.layer3=ResNet.layer3
        self.layer4=ResNet.layer4
        self.avgpool=ResNet.avgpool
        self.dim=512

        self.attibute1 = Local(self.dim, self.dim)
        self.attibute2 = Local(self.dim, self.dim)
        self.attibute3 = Local(self.dim, self.dim)
        self.attibute4 = Local(self.dim, self.dim)
        self.attibute5 = Local(self.dim, self.dim)
        self.attibute6 = Local(self.dim, self.dim)
        self.attibute7 = Local(self.dim, self.dim)
        self.attibute8 = Local(self.dim, self.dim)
        self.attibute9 = Local(self.dim, self.dim)
        self.attibute10 = Local(self.dim, self.dim)

        self.add_attribute=Fusion(dim=self.dim, hdim=self.dim,num_att=self.num_att)
        self.minus_attribute=Fusion(dim=self.dim, hdim=self.dim,num_att=self.num_att)

        self.classifier=nn.Linear(self.dim*(num_att+1),num_classes)
        self.softmax = nn.Softmax(dim=1)

    def compositional_exchange(self,att1,att2):
        ex_index = ((torch.sign(torch.rand(att1.shape[0], self.num_att, 1) - 0.5) + 1) / 2).to(att1.device)
        ex_index = ex_index.repeat(1, 1, 512)
        ex_index = ex_index.view(att1.shape[0], -1)
        att = ex_index * att1 + (1 - ex_index) * att2
        return ex_index,att


    def forward(self, x,same_attribute_prototype=None,different_attribute_prototype=None,label=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_4 = self.layer4(x)
        att1 = self.attibute1(x_4)
        att2 = self.attibute2(x_4)
        att3 = self.attibute3(x_4)
        att4 = self.attibute4(x_4)
        att5 = self.attibute5(x_4)
        att6 = self.attibute6(x_4)
        att7 = self.attibute7(x_4)
        att8 = self.attibute8(x_4)
        att9 = self.attibute9(x_4)
        att10 = self.attibute10(x_4)

        if self.num_att==1:
            att=att1
        elif self.num_att==2:
            att = torch.cat((att1, att2), dim=1)
        elif self.num_att==3:
            att = torch.cat((att1, att2,att3), dim=1)
        elif self.num_att==4:
            att = torch.cat((att1, att2,att3,att4), dim=1)
        elif self.num_att==5:
            att = torch.cat((att1, att2,att3,att4,att5), dim=1)
        elif self.num_att==6:
            att = torch.cat((att1, att2,att3,att4,att5,att6), dim=1)
        elif self.num_att==7:
            att = torch.cat((att1, att2,att3,att4,att5,att6,att7), dim=1)
        elif self.num_att==8:
            att = torch.cat((att1, att2,att3,att4,att5,att6,att7,att8), dim=1)
        elif self.num_att==9:
            att = torch.cat((att1, att2,att3,att4,att5,att6,att7,att8,att9), dim=1)
        elif self.num_att==10:
            att = torch.cat((att1, att2,att3,att4,att5,att6,att7,att8,att9,att10), dim=1)
        # att=att1
        # att=torch.cat((att1,att2),dim=1)
        # att = torch.cat((att1, att2,att3,att4,att5,att6), dim=1)
        x = self.avgpool(x_4)
        x = torch.flatten(x, 1)

        x_cat = torch.cat((x, att), dim=1)
        y = self.classifier(x_cat)


        if same_attribute_prototype is not None:
            same_attribute_prototype = same_attribute_prototype[:, (self.dim):].contiguous()
            different_attribute_prototype = different_attribute_prototype[:, (self.dim):].contiguous()


            # Try different mixing methods
            # compositional exchange
            _, att_ce = self.compositional_exchange(att, same_attribute_prototype)
            ce_x_cat = torch.cat((x, att_ce), dim=1)
            ce_y = self.classifier(ce_x_cat)

            # add attribute
            att_aa = self.add_attribute(att, same_attribute_prototype, aa=True)
            aa_x_cat = torch.cat((x, att_aa), dim=1)
            aa_y = self.classifier(aa_x_cat)

            # mixed attribute
            ex_index,ex_prototype=self.compositional_exchange(same_attribute_prototype,different_attribute_prototype)
            att_mixed=self.minus_attribute(att,ex_prototype)
            mixed_x_cat=torch.cat((x,att_mixed),dim=1)
            mixed_y=self.classifier(mixed_x_cat)

            return x_cat,y,mixed_y,ex_index

        return x_cat,y

