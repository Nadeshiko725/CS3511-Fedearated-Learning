from ast import main
import random
import socket
import numpy as np
import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from io import BytesIO

import tqdm
from model import Net, MLP, BloodMNISTNet
import dill
import torch.nn.functional as F
import argparse
from client import client_run 
from multiprocessing import Process


# MODEL_PATH = './models'
# CLIENT_MODEL_PATH = './models/client_models'
# GLOBAL_MODEL_PATH = './models/global_model.pth'
CLIENT_DATA_PATH = './client_data'
DATA_PATH = './data'
CLIENT_LOG_PATH = './client_log/stage3'
RESULT_PATH = './results/stage3'

def handle_client(client_socket, global_model):
    try:
        # receive data from client
        data = b''  # initialize data as binary string
        while True:
            packet = client_socket.recv(1024)
            if not packet:
                break
            data += packet

        # load data from binary string
        buffer = BytesIO(data)
        buffer.seek(0)  # move the cursor to the beginning
        client_params = torch.load(buffer)

        # load client model
        client_model = BloodMNISTNet()
        client_model.load_state_dict(client_params)

        # do aggregation
        for global_param, client_param in zip(global_model.parameters(), client_model.parameters()):
            global_param.data += client_param.data
    finally:
        # close the connection
        client_socket.close()

def receive_models(global_model, recerive_port=12345, num_clients=20):
    # this function is used to receive models from clients

    # create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # bind the socket to the port
    server_address = ('localhost', recerive_port)
    server_socket.bind(server_address)

    # listen for incoming connections
    server_socket.listen(num_clients)
    for i in range(num_clients):
        # accept a new connection
        print('Waiting for connection for RECEIVING ...')
        client_socket, client_address = server_socket.accept()
        print(f'Connection from {client_address}')
        # create a new thread to handle the connection
        handle_client(client_socket, global_model)

    # close the server socket
    server_socket.close()

def send_models(global_model, client_id, send_port=12346, num_clients=20):
    # this function is used to send the global model to clients

    # create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # bind the socket to the port
    server_address = ('localhost', send_port)
    server_socket.bind(server_address)

    # listen for incoming connections
    server_socket.listen(num_clients)
    for i in range(num_clients):
        # accept a new connection
        print('Waiting for connection for SENDING global_model...')
        client_socket, client_address = server_socket.accept()
        print(f'Connection from {client_address}')

        # send the global model to the client
        buffer = BytesIO()
        model_params = {'client_id': client_id[i], 'model': global_model.state_dict()}
        torch.save(model_params, buffer)
        buffer.seek(0)
        client_socket.sendall(buffer.getvalue())

        # close the connection
        client_socket.close()

    # close the server socket
    server_socket.close()

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

    # test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    # accuracy = correct / total
    return test_loss, accuracy

def server_run(num_epoch=50, num_clients=20, local_rounds=20, lr=0.001, receive_port=12345, send_port=12346):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    global_model = BloodMNISTNet().to(device)

    # zero the parameters of the global model
    for param in global_model.parameters():
        param.data = torch.zeros_like(param.data)
    
    for idx in tqdm.tqdm(range(num_epoch), desc='Epoch', colour='blue'):
        # reecord the cklient number which are connected
        connexted_clients_num = 0

        # receive models from clients
        receive_models(global_model, recerive_port=12345, num_clients=num_clients)

        # do aggregation
        for param in global_model.parameters():
            param.data /= num_clients

        # load the test data
        with open(os.path.join(CLIENT_DATA_PATH, 'Test.pkl'), 'rb') as f:
            test_dataset = dill.load(f)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # evaluate the global model
        test_loss, accuracy = test(global_model, test_loader)
        print(f'Epoch {idx+1}, Test Loss: {test_loss}, Accuracy: {accuracy}')
        with open(os.path.join(RESULT_PATH, 'server_log.txt'), 'a') as f:
            f.write(f'Epoch {idx+1}, Test Loss: {test_loss}, Accuracy: {accuracy}\n')
        
        if num_clients == 20:
            client_id = list(range(1, num_clients+1))
        else:
            client_id = np.random.choice(range(1, 21), num_clients, replace=False)
        
        # send the global model to clients
        send_models(global_model, client_id, send_port=12346, num_clients=num_clients)

if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=50, help='The number of epochs.')
    parser.add_argument('--num_clients', type=int, default=20, help='The number of clients.')
    parser.add_argument('--local_rounds', type=int, default=20, help='The number of local rounds.')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate.')
    parser.add_argument('--receive_port', type=int, default=12377, help='The port for receiving models.')
    parser.add_argument('--send_port', type=int, default=12378, help='The port for sending models.')
    args = parser.parse_args()
    print('Server parse the arguments success')

    server_run(num_epoch=args.num_epoch, num_clients=args.num_clients, local_rounds=args.local_rounds, lr=args.lr, receive_port=args.receive_port, send_port=args.send_port)
    