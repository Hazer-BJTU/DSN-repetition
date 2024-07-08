import module
import load_data
import argparse
import matplotlib.pyplot as plt
from Utility.GPU import try_gpu
import torch


def evaluate_accuracy(net, test_iter_list, device):
    net.eval()
    test_acc, cnt = 0, 0
    predict_list, ground_truth = [], []
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
                predict_list.append(y_hat.argmax(axis=1).item())
                ground_truth.append(y.item())
                cnt += 1
    return float(test_acc / cnt), predict_list, ground_truth


def depict(predict_list, ground_truth):
    figure = plt.figure(figsize=(16, 8))
    figure.canvas.manager.set_window_title('Comparasion')
    expert_fig = figure.add_subplot(2, 1, 1)
    expert_fig.set_title('Ground truth')
    predict_fig = figure.add_subplot(2, 1, 2)
    predict_fig.set_title('Prediction')
    X = list(range(len(predict_list)))
    expert_fig.plot(X, ground_truth)
    predict_fig.plot(X, predict_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose device')
    parser.add_argument('--cuda_idx', type=int, nargs='?', default=0)
    parser.add_argument('--subject', type=int, nargs=1)
    parser.add_argument('--channel', type=str, nargs=1)
    parser.add_argument('--model', type=str, nargs=1)
    parser.add_argument('--sample_rate', type=int, nargs='?', default=100)
    args = parser.parse_args()
    device = try_gpu(args.cuda_idx)
    net = torch.load('./Pretrain Model Save/' + args.model[0], map_location=device)
    data_iter = load_data.load_data_subject(args.subject, args.channel[0],
                                            args.sample_rate, 1, False)
    test_acc, predict_list, ground_truth = evaluate_accuracy(net, [data_iter], device)
    depict(predict_list, ground_truth)
    print(f'Accuracy: {test_acc:.3f}')
    plt.show()
