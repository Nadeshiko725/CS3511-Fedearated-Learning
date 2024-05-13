import os
from pickle import GLOBAL
from pydoc import cli
from shlex import join
import sys
from tabnanny import check
import time
from pyparsing import col
import tqdm
import random
import numpy as np
import pickle
import dill
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from model import Net, MLP, BloodMNISTNet

MODEL_PATH = './models'
CLIENT_MODEL_PATH = './models/client_models'
GLOBAL_MODEL_PATH = './models/global_model.pth'
CLIENT_DATA_PATH = './client_data'
DATA_PATH = './data'
CLIENT_LOG_PATH = './client_log'

def load_data():
    # load client datasets
    train_datasets = []
    for i in range(20):
        with open(os.path.join(CLIENT_DATA_PATH, f'Client{i+1}.pkl'), 'rb') as f:
            train_datasets.append(dill.load(f))

    # load test dataset
    with open(os.path.join(CLIENT_DATA_PATH, 'Test.pkl'), 'rb') as f:
        test_dataset = dill.load(f)
    return train_datasets, DataLoader(test_dataset, batch_size=32, shuffle=False)



def test(global_model, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    global_model.eval()
    correct = 0
    test_loss = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = global_model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()
            target = target.long().squeeze(1)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # correct += torch.sum(pred == target.data)
            # total += target.size(0)

    # test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    # accuracy = correct / total
    return test_loss, accuracy



def train(num_epoch, mode='all', local_rounds=20, num_clients=20, batch_size=32, lr=0.001):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load data
    train_datasets, test_loader = load_data()
    dataloader_for_client = []
    for i in range(20):
        dataloader_for_client.append(DataLoader(train_datasets[i], batch_size=32, shuffle=False, drop_last=True))
    
    # load global model and initialize model parameters
    global_model = BloodMNISTNet().to(device)
    torch.save(global_model.state_dict(), GLOBAL_MODEL_PATH)
    
    best_accuracy = 0
    # train
    for epoch in tqdm.tqdm(range(num_epoch), desc='Epoch', colour='blue'):
        if( mode == 'all'):
            index = [i for i in range(num_clients)]
        else:
            index = random.choices([i for i in range(20)], k=num_clients)
            # index = [0] # for debugging

        for i in tqdm.tqdm(index, desc='Client', colour='green'):
            # load client model
            client_model = BloodMNISTNet().to(device)
            # load global model
            checkpoint = torch.load(GLOBAL_MODEL_PATH)
            client_model.load_state_dict(checkpoint)
            
            # load the dataloader for the client i
            train_loader = dataloader_for_client[i]
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(client_model.parameters(), lr)

            # train client model
            for round in tqdm.tqdm(range(local_rounds), desc='Local Round', colour='yellow'):
                for features, labels in train_loader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    output = client_model(features)
                    labels = labels.long().squeeze(1) # 将 labels 转换为一维
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                # print(f'round {round} Train Loss: {loss.item()}')
                test_loss, accuracy = test(client_model, test_loader)
                # print(f'round {round} Test accuracy: {accuracy}')
                # print(f'round {round} Test loss: {test_loss}')
                # write the loss to the file
                with open (os.path.join(CLIENT_LOG_PATH, f'/results_{num_clients}_{mode}_client{i+1}_log.txt'), 'a') as f:
                    f.write(f'Epoch {epoch}, Client {i+1}, Round {round}, Train Loss: {loss.item()}\n')
                    f.write(f'Epoch {epoch}, Client {i+1}, Round {round}, Test accuracy: {accuracy}\n')
                    f.write(f'Epoch {epoch}, Client {i+1}, Round {round}, Test loss: {test_loss}\n')

            # save client model
            # print(f'Saving client model {i+1}')
            # write the this to file
            # with open(f'./results_{num_clients}_{mode}.txt', 'a') as f:
                # f.write(f'Epoch {epoch}, Saving client model {i+1}\n')
            torch.save(client_model.state_dict(), os.path.join(CLIENT_MODEL_PATH, f'client{i+1}.pth'))
        
        
        # do aggregation process
        aggregation_model = BloodMNISTNet().to(device)
        checkpoint = torch.load(GLOBAL_MODEL_PATH)
        aggregation_model.load_state_dict(checkpoint)

        for i in tqdm.tqdm(index, desc='Aggregation', colour='red'):
            client_model = BloodMNISTNet().to(device)
            checkpoint = torch.load(os.path.join(CLIENT_MODEL_PATH, f'client{i+1}.pth'))
            client_model.load_state_dict(checkpoint)
            for aggregation_param, client_param in zip(aggregation_model.parameters(), client_model.parameters()):
                if i == index[0]:
                    aggregation_param.data = client_param.data
                else:
                    aggregation_param.data += client_param.data
        
        # get the average of the parameters
        for aggregation_param in aggregation_model.parameters():
            aggregation_param.data /= len(index)

        # update the global model
        global_model.load_state_dict(aggregation_model.state_dict())
        
        # global_model.load_state_dict(client_model.state_dict())
        # evaluate
        test_loss, accuracy = test(global_model, test_loader)
        print(f'Epoch {epoch}, Test accuracy: {accuracy}')
        print(f'Eopch {epoch}, Test loss: {test_loss}')


        # save the evaluation results
        with open(f'./results_{num_clients}_{mode}.txt', 'a') as f:
            f.write(f'Epoch {epoch}, Test accuracy: {accuracy}, Test loss: {test_loss}\n')

        if(accuracy > best_accuracy):
            best_accuracy = accuracy
            torch.save(global_model.state_dict(), GLOBAL_MODEL_PATH)


def main():
    # set random seed
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    seed = 0
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if multi-GPU
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled=False
    # make directory
    if not os.path.exists(MODEL_PATH):
        print('Creating model directory')
        os.makedirs(MODEL_PATH)
    if not os.path.exists(CLIENT_MODEL_PATH):
        print('Creating client model directory')
        os.makedirs(CLIENT_MODEL_PATH)
    if not os.path.exists(CLIENT_DATA_PATH):
        print('Creating client data directory')
        os.makedirs(CLIENT_DATA_PATH)
    if not os.path.exists(CLIENT_LOG_PATH):
        print('Creating client log directory')
        os.makedirs(CLIENT_LOG_PATH)
    

    # # stage 1 activate all the 20 clients
    # print('Stage 1: Training all the 20 clients')
    # train(num_epoch=50, mode='all', local_rounds=20, num_clients=20, batch_size=32, lr=0.01)

    # stage 2 activate 10 clients
    print('Stage 2: Training 10 clients randomly')
    train(num_epoch=50, mode='partial', local_rounds=20, num_clients=10, batch_size=32, lr=0.001)

if __name__ == '__main__':
    main()