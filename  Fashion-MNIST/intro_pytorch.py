import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



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
    
    
    train_set = datasets.FashionMNIST('./data', train = True, download=True, transform=transform)
    test_set = datasets.FashionMNIST('./data', train=False, transform=transform)
    
    # set the batch_size = 64 for both train loader and test loader. Besides, set shuffle=False for the test loader
    if training:
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True)
    else:
        loader = torch.utils.data.DataLoader(test_set, batch_size = 64)
    
    return loader
    



def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    
    model = nn.Sequential(
        nn.Flatten(),
        # because one image in the MNIST is 28 * 28  pixel 
        # A Dense layer with 128 nodes and a ReLU activation
        nn.Linear(784, 128),
        nn.ReLU(),
        # A Dense layer with 64 nodes and a ReLU activation
        nn.Linear(128, 64),
        nn.ReLU(),
        # A Dense layer with 10 nodes
        nn.Linear(64, 10)
    )
    return model



"""
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
"""
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
    # let model know this is train
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
    
    # need to run the loop during the epoch 
    for epoch in range(T):
        total_data = 0
        total_loss = 0
        total_correct = 0
        for data in train_loader:
            images, labels = data
            
            #zero the parameter gradients - because in every epoch, the gradient
            # is stored so we need to reset it no every loop
            opt.zero_grad()
            # assign it's output
            outputs = model(images)
            #calculate loss with criterion -> output will be avg loss for number of batch_size image
            loss = criterion(outputs, labels)
            # then we need to check gradient of loss function
            loss.backward()
            # with checking update each parameter for smallest loss
            opt.step()
            
            
            # then we need to get highes result of output
            _, predicted = torch.max(outputs.data,1)
            # add each number of data to total_data
            # because we transform data to 1D array, size(0) is number of data
            total_data += labels.size(0)
            # Then we need to check the number of outputs that is same with label in 64 datas
             # and cast it to python integer
            total_correct +=  (predicted == labels).sum().item()
            # then we need to keep add total loss of 64 data not avg 
            # the reason why use image.size(0) is we are not sure that last input is 64 because 
            # 60000 is not multiple of 64
            total_loss += loss * images.size(0)
        
        # now we need to calculate ouput info
        # calculate the percentage of accuracy
        accuracy = (total_correct / total_data) * 100
        # calculate percentage of loss
        avg_loss = total_loss / total_data
        
            
        # now print out the result
        print(f'Train Epoch: {epoch}\tAccuracy: {total_correct}/{total_data}({accuracy:.2f}%) Loss: {avg_loss:.3f}')
    
    

"""
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
"""
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
    # let model know this is evaluation mode
    model.eval()
    
    #this is evaluation mode so we don't need to track gradients
    with torch.no_grad():
        total_data = 0
        total_loss = 0
        total_correct = 0
        for data in test_loader:
            # assign each data and it's label from data in test_loader
            images, labels = data
            # assign it's output
            outputs = model(images)
            #calculate loss with criterion -> output will be avg loss for number of batch_size image
            loss = criterion(outputs, labels)
            
            # then we need to get highes result of output (1)
            _, predicted = torch.max(outputs.data,1)
            # add each number of data to total_data
            # because we transform data to 1D array, size(0) is number of data
            total_data += labels.size(0)
            # Then we need to check the number of outputs that is same with label in 64 datas 
            # and cast it to python integer
            total_correct +=  (predicted == labels).sum().item()
            # then we need to keep add total loss of 64 data not avg 
            # the reason why use image.size(0) is we are not sure that last input is 64 because 
            # 60000 is not multiple of 64
            total_loss += loss.item() * images.size(0)
        
        # now we need to calculate ouput info
        # calculate the percentage of accuracy
        accuracy = (total_correct / total_data) * 100
        # calculate percentage of loss
        avg_loss = total_loss / total_data
        
        
        # if show_loss is set to False, show only accuracy otherwise show accuracy and Average loss
        if(show_loss == False):
            print(f'Accuracy: {accuracy:.2f}%')
        else:
            print(f'Average loss: {avg_loss:.4f}')
            print(f'Accuracy: {accuracy:.2f}%')
            
            
            
    

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
    
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    
    # we need to get the output of machine with input of single image and 
    # the single image is one iamge in test_images
    output = model(test_images[index])
    
    prob = F.softmax(output,dim = -1)
    
    # change the first dimension which is contained probabilities to list 
    probList = prob[0].tolist()
    
    # create empty dictionary for pairing key will be class and value will be output
    pairDic = {}
    # do mapping probabilities to each class
    for i in range (len(class_names)):
        pairDic[class_names[i]] = probList[i]

    # sorted with descending order
    list_after_sorted = sorted(pairDic.items(),  key=lambda x: x[1], reverse=True)
    
    # get top 3 class and print it
    for rank in range(3):
        class_name, probability = list_after_sorted[rank]
        percent_probability = probability * 100
        print(f"{class_name}: {percent_probability:.2f}%")
        
        
    
    


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
