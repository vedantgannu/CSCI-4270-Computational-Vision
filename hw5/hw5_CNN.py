#!/usr/bin/env python

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.io import read_image
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize
from sklearn.metrics import confusion_matrix
#from google.colab import drive, files
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def prepare_dataset(train_valid_test_option, resize_shape=None, batch_size=32, shuffle_data=False):
    #Aggregate the data
    ROOTPATH = "."
    grass_annotations = pd.DataFrame(data = {"image": os.listdir(os.path.join(ROOTPATH, train_valid_test_option, "grass"))
                                            ,"label":len(os.listdir(os.path.join(ROOTPATH, train_valid_test_option, "grass")))*[0]})
    ocean_annotations = pd.DataFrame(data = {"image": os.listdir(os.path.join(ROOTPATH, train_valid_test_option, "ocean"))
                                            ,"label":len(os.listdir(os.path.join(ROOTPATH, train_valid_test_option, "ocean")))*[1]})
    redcarpet_annotations = pd.DataFrame(data = {"image": os.listdir(os.path.join(ROOTPATH, train_valid_test_option, "redcarpet"))
                                            ,"label":len(os.listdir(os.path.join(ROOTPATH, train_valid_test_option, "redcarpet")))*[2]})
    road_annotations = pd.DataFrame(data = {"image": os.listdir(os.path.join(ROOTPATH, train_valid_test_option, "road"))
                                            ,"label":len(os.listdir(os.path.join(ROOTPATH, train_valid_test_option, "road")))*[3]})
    wheatfield_annotations = pd.DataFrame(data = {"image": os.listdir(os.path.join(ROOTPATH, train_valid_test_option, "wheatfield"))
                                            ,"label":len(os.listdir(os.path.join(ROOTPATH, train_valid_test_option, "wheatfield")))*[4]})

    grass_dataset = CustomDataset(
            img_dir = os.path.join(ROOTPATH, train_valid_test_option, "grass")
            , annotations = grass_annotations
            , transform = Resize(resize_shape) if resize_shape else None
    )
    ocean_dataset = CustomDataset(
            img_dir = os.path.join(ROOTPATH, train_valid_test_option, "ocean")
            , annotations = ocean_annotations
            , transform = Resize(resize_shape) if resize_shape else None
    )
    redcarpet_dataset = CustomDataset(
            img_dir = os.path.join(ROOTPATH, train_valid_test_option, "redcarpet")
            , annotations = redcarpet_annotations
            , transform = Resize(resize_shape) if resize_shape else None
    )
    road_dataset = CustomDataset(
            img_dir = os.path.join(ROOTPATH, train_valid_test_option, "road")
            , annotations = road_annotations
            , transform = Resize(resize_shape) if resize_shape else None
    )
    wheatfield_dataset = CustomDataset(
            img_dir = os.path.join(ROOTPATH, train_valid_test_option, "wheatfield")
            , annotations = wheatfield_annotations
            , transform = Resize(resize_shape) if resize_shape else None
    )
    aggregate_dataset = ConcatDataset([grass_dataset, ocean_dataset, redcarpet_dataset ,road_dataset, wheatfield_dataset])
    final_dataloader = DataLoader(aggregate_dataset, batch_size=batch_size, shuffle=shuffle_data)
    return final_dataloader

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    batches = len(dataloader)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).float(), y.to(device).to(torch.int64)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        #print(train_loss)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(loss)
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return train_loss/batches

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    pred_out = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device).float(), y.to(device).to(torch.int64)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()#Accumulating the loss per batch
            predicted_class = pred.argmax(1)
            correct += (predicted_class == y).type(torch.float).sum().item()
            compare = torch.cat((torch.reshape(predicted_class, (len(predicted_class), 1))
                                                          , torch.reshape(y, (len(y), 1))), 1)
            pred_out.append(compare)
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, torch.cat(pred_out, 0)

#Retreiving datasets for NN training and validation set testing
training_dataloader = prepare_dataset("train", resize_shape=None, batch_size=32, shuffle_data=True)
print("Training Dataloader initialized:", len(training_dataloader.dataset))
validation_dataloader = prepare_dataset("valid", resize_shape=None, batch_size=32, shuffle_data=True)
print("Validation Dataloader initialized: ", len(validation_dataloader.dataset))
testing_dataloader = prepare_dataset("test", resize_shape=None, batch_size=32, shuffle_data=True)
print("Testing Dataloader initialized: ", len(testing_dataloader.dataset))

def model1():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    class NeuralNetwork_Conv(nn.Module):
        def __init__(self):
            super(NeuralNetwork_Conv, self).__init__()
            #Images are R=240 by C=360
            self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            )
            
            #  Resulting image should be 60*90
            self.fc_stack = nn.Sequential(
                nn.Flatten(),
                #Padding preserves shape, but 2 max pools divides dims by 4
                nn.Linear(60*90*32, 128),
                nn.ReLU(),
                nn.Linear(128, 5)
            )

        def forward(self, x):
            logits = self.fc_stack(self.conv_stack(x))
            return logits
        
    #Epochs = 50, Learning Rate = 1e-5
    model_50_1e_5 = NeuralNetwork_Conv().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_50_1e_5.parameters(), lr=1e-5)

    epochs = 50
    validation_losses = []
    training_losses = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        print("Training:")
        training_losses.append(train(training_dataloader, model_50_1e_5, loss_fn, optimizer))
        print("Validation:")
        validation_losses.append(test(validation_dataloader, model_50_1e_5, loss_fn)[0])
    print("Done!")
    plt.plot(np.linspace(1, epochs, epochs), validation_losses, marker='o', color='r', label="Validation")
    plt.plot(np.linspace(1, epochs, epochs), training_losses, marker='o', color='b', label="Training")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.show()

    #Save this model to CNN_50_1e_5.pth
    torch.save(model_50_1e_5.state_dict(), "./CNN_50_1e_5.pth")




def model2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    class NeuralNetwork_Conv(nn.Module):
        def __init__(self):
            super(NeuralNetwork_Conv, self).__init__()
            #Images are R=240 by C=360
            self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
            
            #  Resulting image should be 60*90
            self.fc_stack = nn.Sequential(
                nn.Flatten(),
                #Padding preserves shape, but 2 max pools divides dims by 4
                nn.Linear(60*90*32, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 5)
            )

        def forward(self, x):
            logits = self.fc_stack(self.conv_stack(x))
            return logits

    #Epochs = 50, Learning Rate = 1e-5, 3 input -> 32 featueres, 32 input -> 64 features, 64 -> 32 features, additional hidden layer for NN
    model_50_1e_5_2 = NeuralNetwork_Conv().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_50_1e_5_2.parameters(), lr=1e-5)

    epochs = 50
    validation_losses = []
    training_losses = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        print("Training:")
        training_losses.append(train(training_dataloader, model_50_1e_5_2, loss_fn, optimizer))
        print("Validation:")
        validation_losses.append(test(validation_dataloader, model_50_1e_5_2, loss_fn)[0])
    print("Done!")
    plt.plot(np.linspace(1, epochs, epochs), validation_losses, marker='o', color='r', label="Validation")
    plt.plot(np.linspace(1, epochs, epochs), training_losses, marker='o', color='b', label="Training")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.show()

    #Save this model to CNN_50_1e_5_2.pth
    torch.save(model_50_1e_5_2.state_dict(), "./CNN_50_1e_5_2.pth")


def model3():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    class NeuralNetwork_Conv(nn.Module):
        def __init__(self):
            super(NeuralNetwork_Conv, self).__init__()
            #Images are R=240 by C=360
            self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
            
            #  Resulting image should be 60*90
            self.fc_stack = nn.Sequential(
                nn.Flatten(),
                nn.Linear(60*90*64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 5)
            )

        def forward(self, x):
            logits = self.fc_stack(self.conv_stack(x))
            return logits

    #Epochs = 50, Learning Rate = 1e-5, 3 input -> 32 featueres, 32 input -> 64 features, 64 -> 128 features, 128 -> 64 features
    #with additional hidden layer from NN 2
    model_50_1e_5_3 = NeuralNetwork_Conv().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_50_1e_5_3.parameters(), lr=1e-5)

    epochs = 50
    validation_losses = []
    training_losses = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        print("Training:")
        training_losses.append(train(training_dataloader, model_50_1e_5_3, loss_fn, optimizer))
        print("Validation:")
        validation_losses.append(test(validation_dataloader, model_50_1e_5_3, loss_fn)[0])
    print("Done!")
    plt.plot(np.linspace(1, epochs, epochs), validation_losses, marker='o', color='r', label="Validation")
    plt.plot(np.linspace(1, epochs, epochs), training_losses, marker='o', color='b', label="Training")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.show()

    #Save this model to CNN_50_1e_5_3.pth
    torch.save(model_50_1e_5_3.state_dict(), "./CNN_50_1e_5_3.pth")


def bestmodel():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    class NeuralNetwork_Conv(nn.Module):
        def __init__(self):
            super(NeuralNetwork_Conv, self).__init__()
            #Images are R=240 by C=360
            self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
            
            #  Resulting image should be 60*90
            self.fc_stack = nn.Sequential(
                nn.Flatten(),
                nn.Linear(60*90*64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 5)
            )

        def forward(self, x):
            logits = self.fc_stack(self.conv_stack(x))
            return logits

    #Applying CNN_50_1e_5_3 to testing data
    model_50_1e_5_3_loaded = NeuralNetwork_Conv().to(device)

    model_50_1e_5_3_loaded.load_state_dict(torch.load("./CNN_50_1e_5_3.pth"
                                        ,map_location=device))
    loss_fn = nn.CrossEntropyLoss()

    _ , pred_vs_actual = test(testing_dataloader, model_50_1e_5_3_loaded, loss_fn)
    print(confusion_matrix(pred_vs_actual.cpu().numpy()[:,1], pred_vs_actual.cpu().numpy()[:,0]))

if __name__ == "__main__":
    model1()
    model2()
    model3()
    bestmodel()
