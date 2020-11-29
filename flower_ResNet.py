import os
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms,datasets
import matplotlib.pyplot as plt



__all__ = ['ResNet50', 'ResNet101','ResNet152']

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
            nn.Dropout(p=0.7)
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=21, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048,num_classes)
        self.drop_layer=nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet50():
    return ResNet([3, 4, 6, 3])

def ResNet101():
    return ResNet([3, 4, 23, 3])

def ResNet152():
    return ResNet([3, 8, 36, 3])

 
def train_accuracy(model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in train_loader:
            img, labels = data
            img=img.to(device)
            labels=labels.to(device)
            model=model.to(device)
            out = model(img)
            _,pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print('Accuracy of the network on the train image: %d %%' % (100 * correct / total))
    return 100.0 * correct / total
 
def test_accuracy(model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            img, labels = data
            img=img.to(device)
            labels=labels.to(device)
            model=model.to(device)
            out = model(img)
            _,pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print('Accuracy of the network on the test image: %d %%' % (100 * correct / total))
    return 100.0 * correct / total
 
 
 
def train(model):
    #定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    print(device)
    model=model.to(device)
    optimizer = optim.SGD(model.parameters(), lr = LR, momentum=0.9, weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
#    optimizer = optim.SGD(model.parameters(), lr = LR, momentum=0.9)
#    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99),weight_decay=5e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=12, gamma = 0.5)
#    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    iter = 0
    num = 1
    loss_list=[]
    train_accuracy_list=[]
    test_accuracy_list=[]
    #训练网络
    for epoch in range(num_epoches):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            iter = iter + 1
            img, labels = data
            img=img.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            #训练
            model=model.to(device)
            out = model(img)
            loss = criterion(out, labels)
            if i%10==0:
               print(i,':',loss.item())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
 
        scheduler.step() 
        loss_epoch=running_loss/(batchSize * (i + 1))
        loss_list.append(loss_epoch)
        
        
        print('epoch: %d\t batch: %d\t lr: %g\t loss: %.6f' % (epoch + 1, i + 1, scheduler.get_lr()[0],loss_epoch))
        print('\n')
        num = num + 1
 
        torch.save(model, './flower_classification_model.pkl')
        
        if epoch%5==0:
            train_acc=train_accuracy(model)
            test_acc=test_accuracy(model)
            train_accuracy_list.append(train_acc)
            test_accuracy_list.append(test_acc)
        
    x_data=range(num_epoches)
    y_data1=loss_list
    y_data2=train_accuracy_list
    y_data3=test_accuracy_list
    plt.figure(figsize=(15,18))
    plt.subplot(2,1,1)
    plt.plot(x_data,y_data1,color='red',linewidth=2.0,linestyle='--',label='loss_curve')
    plt.legend()
    plt.xlabel('epoches')
    plt.ylabel('loss')
   
    plt.subplot(2,1,2)
    plt.plot(x_data,y_data2,color='yellow',linewidth=2.0,linestyle='-.',label='train_acc')
    plt.plot(x_data,y_data3,color='blue',linewidth=2.0,linestyle='-.',label='test_acc')
    plt.legend()
    plt.xlabel('epoches')
    plt.ylabel('acc')
    
    plt.savefig('./plot.jpg')
    plt.show()
    

 
if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modelPath = './flower_classification_model.pkl'
    batchSize = 16
    LR = 0.01
    num_epoches = 45
    

    train_data = datasets.ImageFolder(r'/home/fengbuding/Desktop/flowers/train_data',transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),transforms.Resize((224,224)),transforms.ToTensor(),
    transforms.Normalize((0.32668, 0.41849, 0.39262),(0.30282, 0.29514, 0.29262))]))
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batchSize,shuffle=True)

    test_data = datasets.ImageFolder(r'/home/fengbuding/Desktop/flowers/test_data',transform=transforms.Compose([
    transforms.Resize((224,224)),transforms.ToTensor(),
    transforms.Normalize((0.32668, 0.41849, 0.39262),(0.30282, 0.29514, 0.29262))]))
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=batchSize,shuffle=True)
    
    model=ResNet50()
    if os.path.exists(modelPath):
        print('model exists')
        model=torch.load(modelPath)
        print('model load')
    else:
        print('model not exists')
        print('Training starts')
        train(model)
        print('Training Finished')