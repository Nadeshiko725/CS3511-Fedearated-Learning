from struct import pack
from pyparsing import C
import torch
import os
import sys
import numpy as np
import random
import dill
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from io import BytesIO

import tqdm
from model import Net, MLP, BloodMNISTNet
from io import BytesIO
import socket
import argparse
import time

CLIENT_DATA_PATH = './client_data'
DATA_PATH = './data'
CLIENT_LOG_PATH = './client_log/stage3'
RESULT_PATH = './results/stage3'

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
            target = target.long().squeeze(1)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy

def train_and_sendback(client_id, num_epochs, local_round, lr, server_ip, receive_port, send_port):
    # initialize model
    client_model = BloodMNISTNet()

    # for epoch in range(num_epochs):
    desc = f'Client {client_id} Training'

    for epoch in tqdm.tqdm(range(num_epochs), desc=desc, colour='blue'):
        # load the global model parameters
        if epoch > 0:
            client_id, global_params = receive_global_params(server_ip, receive_port)
            client_model.load_state_dict(global_params)

        with open(os.path.join(CLIENT_DATA_PATH, f'Client{client_id}.pkl'), 'rb') as f:
            train_dataset = dill.load(f)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

        # train the model for local_round 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)

        desc = f'Epoch {epoch}, Client {client_id} Local Round'
        for local_epoch in tqdm.tqdm(range(local_round), desc=desc, colour='yellow'):
            for feature, labels in train_loader:
                feature, labels = feature.to(device), labels.to(device)
                optimizer.zero_grad()
                output = client_model(feature)
                labels = labels.long().squeeze(1)   # squeeze the labels to remove the extra dimension
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
            test_loss, accuracy = test(client_model, train_loader)
            with open(os.path.join(CLIENT_LOG_PATH, f'Client{client_id}.txt'), 'a') as f:
                f.write(f'Epoch {epoch}, Local Round {local_epoch}, Loss: {test_loss}, Accuracy: {accuracy}\n')

        # send back the model parameters
        send_model_params(client_model, server_ip, send_port)

def send_model_params(client_model, server_ip, server_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            s.connect((server_ip, server_port))
            break
        except ConnectionRefusedError:
            print('Connection refused. Retrying...')
            time.sleep(1)
    # print('Connected to server')
    buffer = BytesIO()
    torch.save(client_model.state_dict(), buffer)
    buffer.seek(0)
    s.sendall(buffer.getvalue())
    s.close()


def receive_global_params(server_ip, server_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            s.connect((server_ip, server_port))
            break
        except ConnectionRefusedError:
            print('Connection refused. Retrying...')
            time.sleep(1)

    print('Connected to server')
    params_bytes = b''
    while True:
        packet = s.recv(1024)
        if not packet:
            break
        params_bytes += packet
    buffer = BytesIO(params_bytes)
    buffer.seek(0)
    global_params = torch.load(buffer)
    s.close()
    client_id = global_params['client_id']
    params = global_params['model_state_dict']
    return client_id, params

def client_run(client_id, num_epochs, local_round, lr, server_ip, receive_port, send_port):
    # set the seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    # make directory if not exist
    if not os.path.exists(CLIENT_LOG_PATH):
        os.makedirs(CLIENT_LOG_PATH)
    if not os.path.exists(CLIENT_DATA_PATH):
        os.makedirs(CLIENT_DATA_PATH)
    
    train_and_sendback(client_id, num_epochs, local_round, lr, server_ip, receive_port, send_port)