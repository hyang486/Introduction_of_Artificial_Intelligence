import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
from torchvision import datasets, transforms

from torch.utils.data import DataLoader


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    # test_set = datasets.FashionMNIST('./data', train=False,transform=transform)

    dataset = datasets.FashionMNIST('./data', train=training, download=True, transform=transform)
    return DataLoader(dataset, batch_size=64, shuffle=transform)




def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128, bias=True),
        nn.ReLU(),
        nn.Linear(128, 64, bias=True),
        nn.ReLU(),
        nn.Linear(64, 10)
    )




def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """

    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    model.train()
    for e in range(T):
        correct = 0
        total = 0
        total_loss = 0
        for step, (data, label) in enumerate(train_loader):
            data = data.cuda()
            label = label.cuda()
            
            output = model(data)
            loss = criterion(output, label)

            total += data.size(0)
            correct += torch.count_nonzero(data == label).item()
            total_loss += loss.item() * data.size(0)
    
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f'Train Epoch: {e}\tAccuracy:{correct}/{total}({round(correct/total, 2)}%)\tLoss: {round(total_loss/total, 3)}')
    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.cuda()
            labels = labels.cuda()

            output = model(data)
            loss = criterion(output, labels)

            total += data.size(0)
            correct += torch.count_nonzero(data == label).item()
            total_loss += loss.item() * data.size(0)

    if show_loss:
        print(f'Loss: {round(total_loss/total, 3)}')
        print(f'Accuracy: {round(correct/total, 2)}%')
    else:
        print(f'Accuracy: {round(correct / total, 2)}%')
    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T - shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                   'Shirt', 'Sneaker','Bag','Ankle', 'Boot']

    output = model(test_images)
    prob = F.softmax(output, dim=1)
    dic = {name : value for name, value in zip(class_names, prob)}
    dic = dict(sorted(dic.items(), key=lambda x: x[1], reverse=True))
    for k, v in dic.keys()[:index], dic.values()[:index]:
        print(f'{k}:{round(v, 2)}')


    if __name__ == '__main__':
        '''
        Feel free to write your own test code here to exaime the correctness of your functions. 
        Note that this part will not be graded.
        '''
        criterion = nn.CrossEntropyLoss()

        model = build_model()
        train_loader = get_data_loader(True)
        test_loader = get_data_loader(False)
        train_model(model, train_loader, criterion, 5)
        evaluate_model(model, test_loader, criterion, True)
        # predict_label(model, None, 3)