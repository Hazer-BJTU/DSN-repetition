import torch
import module
import load_data
from torch import nn
from Utility.GPU import try_gpu
from train import accuracy
import argparse
import sys
import train as pre_train


def evaluate_accuracy_gpu(net, test_iter_list, device):
    net.eval()
    test_acc, cnt = 0, 0
    with torch.no_grad():
        for test_iter in test_iter_list:
            H0, C0 = net.get_initial_states(1, device)
            states = (H0, C0)
            for X, y in test_iter:
                X = torch.unsqueeze(X, dim=1)
                X, y = X.to(device), y.to(device)
                y_hat, states = net(X, states, 1)
                if y.item() == y_hat.argmax(axis=1).item():
                    test_acc += 1
                cnt += 1
    return float(test_acc / cnt)


def train(net, train_iter, test_iter_list, num_epochs, lr, device, window_size, weight_decay):
    print('Training on ', device)
    optimizer = torch.optim.SGD([
        {'params': net.rnn_layer.parameters(), 'lr': lr, 'weight_decay': 0.0},
        {'params': net.residual_link.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': net.classifier.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': net.feature_extraction.parameters(), 'lr': lr / 10.0, 'weight_decay': weight_decay}
    ], lr=lr)
    loss = nn.CrossEntropyLoss()
    train_l, train_acc, test_acc = 0, 0, 0
    for epoch in range(num_epochs):
        train_l, train_acc, test_acc, cnt, cnt2 = 0, 0, 0, 0, 0
        net.train()
        for i, (X, y) in enumerate(train_iter):
            cnt += X.shape[0] * X.shape[1]
            cnt2 += X.shape[1]
            H0, C0 = net.get_initial_states(X.shape[0], device)
            X = X.view(-1, 1, 3000)
            y = y.view(-1)
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat, _ = net(X, (H0, C0), window_size)
            L = loss(y_hat, y)
            train_l += L
            train_acc += accuracy(y_hat, y)
            L.backward()
            nn.utils.clip_grad_norm_(net.rnn_layer.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
        train_l /= cnt2
        train_acc /= cnt
        test_acc = evaluate_accuracy_gpu(net, test_iter_list, device)
        print(f'Epoch {epoch} Loss: {train_l:.3f}, Train acc: {train_acc:.3f}, '
              f'Test acc: {test_acc:.3f}')
    return train_l, train_acc, test_acc


def k_fold_train(channel, cuda_idx, num_epochs_pre, num_epochs, lr_pre, lr, weight_decay_pre, weight_decay,
                 batch_size, window_size):
    device = try_gpu(cuda_idx)
    batch_size_pre = 128
    sample_rate = 100
    results = []
    total_train_l, total_train_acc, total_test_acc = 0, 0, 0
    for i in range(5):
        j = 2 * i + 1
        test_subjects = [j]
        train_subjects = list(range(1, 11))
        train_subjects.remove(j)
        train_subjects.remove(j + 1)
        pre_train_iter = load_data.load_data_subject(train_subjects, channel, sample_rate,
                                                     batch_size_pre, True)
        pre_test_iter = load_data.load_data_subject(test_subjects, channel, sample_rate,
                                                    batch_size_pre, True)
        net = module.get_rl_net(sample_rate, 128 * 25, 5)
        pre_train.train(net, pre_train_iter, pre_test_iter, num_epochs_pre, lr_pre, device, weight_decay_pre)
        file_path = './Pretrain Model/FeatureExtraction-' + str(i) + '.pth'
        torch.save(net[0], file_path)
        train_iter = load_data.load_data_subject_sequence(train_subjects, channel, sample_rate,
                                                          batch_size, window_size, True)
        test_iter_list = []
        for subject_idx in test_subjects:
            test_iter_list.append(
                load_data.load_data_subject([subject_idx], channel, sample_rate, 1, False)
            )
        DSN = module.SequenceLearning(file_path, sample_rate, 128 * 25, device)
        train_l, train_acc, test_acc = train(DSN, train_iter, test_iter_list, num_epochs, lr,
                                             device, window_size, weight_decay)
        total_train_l += train_l
        total_train_acc += train_acc
        total_test_acc += test_acc
        results.append((train_l, train_acc, test_acc))
        torch.save(DSN, './Pretrain Model/DSN-' + str(i) + '-' + channel + '.pth')
    with open('output.txt', 'w') as file:
        original_stdout = sys.stdout
        sys.stdout = file
        print(f'(train_l, train_acc, test_acc)')
        for i, result in enumerate(results):
            print(f'Case{i}: ({result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f})')
        print(f'Average train loss: {total_train_l / 5:.3f}, average train acc: {total_train_acc / 5:.3f}, '
              f'average test acc: {total_test_acc / 5:.3f}')
        sys.stdout = original_stdout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose device')
    parser.add_argument('--channel', nargs='?', default='F3_A2')
    parser.add_argument('--cuda_idx', type=int, nargs='?', default=0)
    parser.add_argument('--num_epochs_pre', type=int, nargs='?', default=300)
    parser.add_argument('--num_epochs', type=int, nargs='?', default=95)
    parser.add_argument('--lr_pre', type=float, nargs='?', default=0.001)
    parser.add_argument('--lr', type=float, nargs='?', default=0.005)
    parser.add_argument('--weight_decay_pre', type=float, nargs='?', default=0.0015)
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.001)
    parser.add_argument('--batch_size', type=int, nargs='?', default=64)
    parser.add_argument('--window_size', type=int, nargs='?', default=20)
    args = parser.parse_args()
    k_fold_train(args.channel, args.cuda_idx, args.num_epochs_pre, args.num_epochs, args.lr_pre, args.lr,
                 args.weight_decay_pre, args.weight_decay, args.batch_size, args.window_size)
