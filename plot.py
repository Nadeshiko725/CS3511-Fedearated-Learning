from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

RESULT_PATH = './results'

def plot_loss_acc():
    # 读入不同client数的loss和acc
    num_clients = [4, 8, 10, 15, 20]
    # 创建两个二维数组，分别存储loss和acc
    loss = []
    acc = []
    for i in range(len(num_clients)):
        with open(os.path.join(RESULT_PATH, f'results_{num_clients[i]}_partial_multi.txt'), 'rb') as f:
            lines = f.readlines()
            # 略过第一行
            lines = lines[1:]
            # 保存num_clients为num_clients[i]时的loss和acc
            loss.append([])
            acc.append([])
            for line in lines:
                line = line.decode('utf-8')
                loss[i].append(float(line.split(',')[2].split(':')[1]))
                acc[i].append(float(line.split(',')[1].split(':')[1]))
    # 绘制loss和acc
    plt.figure()
    for i in range(len(num_clients)):
        plt.plot(loss[i], label=f'{num_clients[i]} clients', linestyle='--')
    plt.title('Loss for different number of clients')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross Entropy)')
    plt.legend()
    plt.savefig(os.path.join(RESULT_PATH, 'loss_different_M.png'))
    plt.show()

    plt.figure()
    for i in range(len(num_clients)):
        plt.plot(acc[i], label=f'{num_clients[i]} clients')
    plt.title('Accuracy for different number of clients')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy for gloabl model')
    plt.legend()
    plt.savefig(os.path.join(RESULT_PATH, 'accuracy_different_M.png'))
    plt.show()
 

def plot_for_stage3():
    num_clients = [5, 10, 15, 20]
    results = {num: {'loss': [], 'acc': []} for num in num_clients}
    with open(os.path.join(RESULT_PATH, 'results_socket.txt'), 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if 'num_clients' in line:
            num = int(line.split('=')[-1].strip())
            for j in range(1, 11):
                round_line = lines[i+j].strip()
                round_loss = float(round_line.split(',')[1].split(':')[-1].strip())
                round_acc = float(round_line.split(',')[2].split(':')[-1].strip())
                results[num]['loss'].append(round_loss)
                results[num]['acc'].append(round_acc)
            i += 11
        else:
            i += 1

    # plot
    plt.figure()
    for num in num_clients:
        plt.plot(results[num]['loss'], label=f'{num} clients', linestyle='--')
    plt.title('Loss for different number of clients')
    plt.xlabel('Epoch')
    plt.ylabel('Loss with different client number')
    plt.gca().get_yaxis().set_visible(False)
    plt.legend()
    plt.savefig(os.path.join(RESULT_PATH,'loss_socket.png'))

    # plot accuracy
    plt.figure()
    for num in num_clients:
        plt.plot(results[num]['acc'], label=f'{num} clients')
    plt.title('Accuracy for different number of clients')
    plt.xlabel('Round')
    plt.ylabel('Accuracy with different client number')
    plt.legend()
    plt.savefig(os.path.join(RESULT_PATH, 'accuracy_socket.png'))


if __name__ == '__main__':
    # plot_loss_acc()
    plot_for_stage3()

        