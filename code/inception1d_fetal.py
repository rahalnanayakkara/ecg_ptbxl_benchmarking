import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision
import math
import numpy as np

from fastai.layers import *
from fastai.core import *

from models.basic_conv1d import AdaptiveConcatPool1d,create_head1d

def conv(in_planes, out_planes, kernel_size=3, stride=1):
    "convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)

def noop(x): return x

class InceptionBlock1d(nn.Module):
    def __init__(self, ni, nb_filters, kss, stride=1, act='linear', bottleneck_size=32):
        super().__init__()
        self.bottleneck = conv(ni, bottleneck_size, 1, stride) if (bottleneck_size>0) else noop

        self.convs = nn.ModuleList([conv(bottleneck_size if (bottleneck_size>0) else ni, nb_filters, ks) for ks in kss])
        self.conv_bottle = nn.Sequential(nn.MaxPool1d(3, stride, padding=1), conv(ni, nb_filters, 1))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d((len(kss)+1)*nb_filters), nn.ReLU())

    def forward(self, x):
        #print("block in",x.size())
        bottled = self.bottleneck(x)
        out = self.bn_relu(torch.cat([c(bottled) for c in self.convs]+[self.conv_bottle(x)], dim=1))
        return out

class Shortcut1d(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.act_fn=nn.ReLU(True)
        self.conv=conv(ni, nf, 1)
        self.bn=nn.BatchNorm1d(nf)

    def forward(self, inp, out): 
        #print("sk",out.size(), inp.size(), self.conv(inp).size(), self.bn(self.conv(inp)).size)
        #input()
        return self.act_fn(out + self.bn(self.conv(inp)))
        
class InceptionBackbone(nn.Module):
    def __init__(self, input_channels, kss, depth, bottleneck_size, nb_filters, use_residual):
        super().__init__()

        self.depth = depth
        assert((depth % 3) == 0)
        self.use_residual = use_residual

        n_ks = len(kss) + 1
        self.im = nn.ModuleList([InceptionBlock1d(input_channels if d==0 else n_ks*nb_filters,nb_filters=nb_filters,kss=kss, bottleneck_size=bottleneck_size) for d in range(depth)])
        self.sk = nn.ModuleList([Shortcut1d(input_channels if d==0 else n_ks*nb_filters, n_ks*nb_filters) for d in range(depth//3)])    
        
    def forward(self, x):
        
        input_res = x
        for d in range(self.depth):
            x = self.im[d](x)
            if self.use_residual and d % 3 == 2:
                x = (self.sk[d//3])(input_res, x)
                input_res = x.clone()
        return x

class Inception1d(nn.Module):
    '''inception time architecture'''
    def __init__(self, num_classes=2, input_channels=8, kernel_size=40, depth=6, bottleneck_size=32, nb_filters=32, use_residual=True,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
        super().__init__()
        assert(kernel_size>=40)
        kernel_size = [k-1 if k%2==0 else k for k in [kernel_size,kernel_size//2,kernel_size//4]] #was 39,19,9
        
        layers = [InceptionBackbone(input_channels=input_channels, kss=kernel_size, depth=depth, bottleneck_size=bottleneck_size, nb_filters=nb_filters, use_residual=use_residual)]
       
        n_ks = len(kernel_size) + 1
        #head
        head = create_head1d(n_ks*nb_filters, nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        layers.append(head)
        #layers.append(AdaptiveConcatPool1d())
        #layers.append(Flatten())
        #layers.append(nn.Linear(2*n_ks*nb_filters, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)
    
    def get_layer_groups(self):
        depth = self.layers[0].depth
        if(depth>3):
            return ((self.layers[0].im[3:],self.layers[0].sk[1:]),self.layers[-1])
        else:
            return (self.layers[-1])
    
    def get_output_layer(self):
        return self.layers[-1][-1]
    
    def set_output_layer(self,x):
        self.layers[-1][-1] = x
    
def inception1d(**kwargs):
    """Constructs an Inception model
    """
    return Inception1d(**kwargs)


n_classes = 4
n_channels = 12
batch_size = 10

inception1d_model = Inception1d(num_classes=n_classes, input_channels=n_channels)

test_input = torch.Tensor(batch_size, n_channels, 256)

test_output = inception1d_model(test_input)

print("Test Input shape : ", test_input.shape)
print("Test Output shape : ", test_output.shape)


X_train = np.load('dataset/Train/X_train.npy')
y_train = np.load('dataset/Train/y_train.npy')
X_test = np.load('dataset/Test/X_test.npy')
y_test = np.load('dataset/Test/y_test.npy')

print("Shapes of training and testing tensors")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device : ", device)


class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.longlong))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


batch_size = 10
number_of_classes = 4

# Instantiate training and test data
train_data = Data(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
print("The number of signals in a training set is: ", len(train_loader)*batch_size)

test_data = Data(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
print("The number of signals in a test set is: ", len(test_loader)*batch_size)

print("The number of batches per epoch is: ", len(train_loader))


from torch.optim import Adam

model = inception1d_model
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


def saveModel(model):
    path = "inception1d_model.pth"
    torch.save(model.state_dict(), path)

def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0
    with torch.no_grad():
        for data in test_loader:
            spectro, labels = data
            spectro = spectro.to(device)
            labels = labels.to(device)
            outputs = model(spectro)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):

    best_accuracy = 0.0

    # Print your execution device
    print("The model will be running on", device, "device")

    # Convert model parameters and buffers to CPU or mps
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0

        for i, (spectro, labels) in enumerate(train_loader, 0):
            spectro = spectro.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # predict classes using spectrograms from the training set
            outputs = model(spectro)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)

            # backpropagate the loss
            loss.backward()

            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Print statistics for every 100 spectrograms
            running_loss += loss.item()     # extract the loss value
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 100))

                # zero the loss
                running_loss = 0.0

        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))

        # Save the model if the accuracy is the best
        if accuracy >= best_accuracy:
            saveModel(model)
            best_accuracy = accuracy


train(50)
print('Finished Training')


