import random
import socket
import subprocess
import numpy as np
import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from io import BytesIO
from model import Net, MLP, BloodMNISTNet
import dill
import torch.nn.functional as F
import argparse
from client import client_run
from multiprocessing import Process


CLIENT_DATA_PATH = './client_data'
DATA_PATH = './data'
CLIENT_LOG_PATH = './client_log/stage3'
RESULT_PATH = './results/stage3'


def main():
    # make directory if not exist
    if not os.path.exists(CLIENT_LOG_PATH):
        os.makedirs(CLIENT_LOG_PATH)
    if not os.path.exists(CLIENT_LOG_PATH):
        os.makedirs(CLIENT_LOG_PATH)
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    print('make directory success')

    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=50, help='The number of epochs.')
    parser.add_argument('--num_clients', type=int, default=20, help='The number of clients.')
    parser.add_argument('--local_rounds', type=int, default=20, help='The number of local rounds.')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate.')
    parser.add_argument('--receive_port', type=int, default=12377, help='The port for receiving models.')
    parser.add_argument('--send_port', type=int, default=12378, help='The port for sending models.')
    args = parser.parse_args()
    print('parse the arguments success')

    # set the random seed
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if multi-GPU
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled=False
    
    # start the server
    print('Stage3 : Training with socket communication.')
    # server_run(num_epoch=args.num_epoch, num_clients=args.num_clients, local_rounds=args.local_rounds, lr=args.lr, receive_port=args.receive_port, send_port=args.send_port)
    server_process = subprocess.Popen(['python', 'server.py', '--num_epoch', str(args.num_epoch), '--num_clients', str(args.num_clients), '--local_rounds', str(args.local_rounds), '--lr', str(args.lr), '--receive_port', str(args.receive_port), '--send_port', str(args.send_port)])

    # start all the clients
    process = []
    for client_id in range(args.num_clients):
        # start a client in a new process
        p = Process(target=client_run, args=(client_id, args.num_epoch, args.local_rounds, args.lr, 'localhost', args.receive_port, args.send_port))
        p.start()
        process.append(p)
    
    # wait for all the clients to finish
    for p in process:
        p.join()
    
    # terminate the server
    server_process.terminate()

if __name__ == '__main__':
    main()
