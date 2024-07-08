import torch
import module
import load_data
from torch import nn
from Utility.Timer import Timer
from Utility.Accumulator import Accumulator
from Utility.GPU import try_gpu
import argparse
import sys


def accuracy(y_hat, y):
    y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat.to(torch.int64) == y)
    return float(cmp.sum().item())


def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X = torch.unsqueeze(X, dim=1)
            y = torch.squeeze(y, dim=1)
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train(net, train_iter, test_iter, num_epochs, lr, device, weight_decay):
    print('Training on ', device)
    net.to(device)
    '''optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.001)'''
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss()
    '''
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    '''
    timer, num_batches = Timer(), len(train_iter)
    metric = Accumulator(3)
    train_l, train_acc, test_acc = 0, 0, 0
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            X = torch.unsqueeze(X, dim=1)
            y = torch.squeeze(y, dim=1)
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            L = loss(y_hat, y)
            L.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(L * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            '''
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
            '''
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        '''animator.add(epoch + 1, (None, None, test_acc))'''
        print(f'{epoch / num_epochs * 100:.1f}% work complete...')
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')
    return train_l, train_acc, test_acc


def k_fold_train(channel, cuda_idx, num_epochs, lr, weight_decay):
    batch_size = 128
    sample_rate = 100
    results = []
    total_train_l, total_train_acc, total_test_acc = 0, 0, 0
    for i in range(5):
        j = i * 2 + 1
        test_subjects = [j, j + 1]
        train_subjects = list(range(1, 11))
        train_subjects.remove(j)
        train_subjects.remove(j + 1)
        net = module.get_rl_net(sample_rate, 128 * 25, 5)
        train_iter = load_data.load_data_subject(train_subjects, channel, sample_rate, batch_size, True)
        test_iter = load_data.load_data_subject(test_subjects, channel, sample_rate, batch_size, True)
        train_l, train_acc, test_acc = train(net, train_iter, test_iter, num_epochs,
                                             lr, try_gpu(cuda_idx), weight_decay)
        results.append((train_l, train_acc, test_acc))
        total_train_l += train_l
        total_train_acc += train_acc
        total_test_acc += test_acc
        torch.save(net[0], './Pretrain Model/FeatureExtraction-' + str(i) + '.pth')
    with open('output.txt', 'w') as file:
        original_stdout = sys.stdout
        sys.stdout = file
        print('(loss, train acc, test acc)')
        for i, result in enumerate(results):
            print(f'Case {i}: ({result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f})')
        print(f'Average loss {total_train_l / 5:.3f}, Average train acc {total_train_acc / 5:.3f}, '
              f'Average test acc {total_test_acc / 5:.3f}')
        sys.stdout = original_stdout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose device')
    parser.add_argument('--cuda_idx', type=int, nargs='?', default=0)
    parser.add_argument('--num_epochs', type=int, nargs='?', default=250)
    parser.add_argument('--lr', type=float, nargs='?', default=0.001)
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.001)
    parser.add_argument('--channel', nargs='?', default='F3_A2')
    args = parser.parse_args()
    k_fold_train(args.channel, args.cuda_idx, args.num_epochs, args.lr, args.weight_decay)
