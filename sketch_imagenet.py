import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args
import utils.common as utils

import os
import time
from data import imagenet_dali
from importlib import import_module

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

# Data
print('==> Preparing data..')
def get_data_set(type='train'):
    if type == 'train':
        return imagenet_dali.get_imagenet_iter_dali('train', args.data_path, args.train_batch_size,
                                                   num_threads=4, crop=224, device_id=args.gpus[0], num_gpus=1)
    else:
        return imagenet_dali.get_imagenet_iter_dali('val', args.data_path, args.eval_batch_size,
                                                   num_threads=4, crop=224, device_id=args.gpus[0], num_gpus=1)
trainLoader = get_data_set('train')
testLoader = get_data_set('test')

def weight_norm(weight, weight_norm_method=None):

    if weight_norm_method == 'l2':
        norm_func = lambda x: torch.sqrt(torch.sum(x.pow(2)))
    else:
        norm_func = lambda x: 1.0

    weight /= norm_func(weight)

    return weight

def sketch_matrix(weight, l, dim, weight_norm_method=None):

    A = weight.clone()
    if weight.dim() == 4:  #Convolution layer
        A = A.view(A.size(dim), -1)

    B = torch.zeros(l, A.size(1))
    ind = int(l / 2)
    [n, _] = A.size()  # n: number of samples m: dimension
    numNonzeroRows = torch.nonzero(torch.sum(B.mul(B), 1) > 0).size(0) # number of non - zero rows

    for i in range(n):
        if numNonzeroRows < l:
            B[numNonzeroRows, :] = A[i, :]
        else:
            if n - i < l // 2:
                break
            u, sigma, _ = torch.svd(B.t())
            sigmaSquare = sigma.mul(sigma)
            sigmaSquareDiag = torch.diag(sigmaSquare)
            theta = sigmaSquareDiag[ind]
            sigmaSquare = sigmaSquareDiag - torch.eye(l) * torch.sum(theta)
            sigmaHat = torch.sqrt(torch.where(sigmaSquare > 0,
                                              sigmaSquare, torch.zeros(sigmaSquare.size())))
            B = sigmaHat.mm(u.t())
            numNonzeroRows = ind
            B[numNonzeroRows, :] = A[i, :]

        numNonzeroRows = numNonzeroRows + 1

    if dim == 0:
        return weight_norm(B.view(l, weight.size(1), weight.size(2), weight.size(3)), weight_norm_method)
    elif dim == 1:
        return weight_norm(B.view(weight.size(0), l, weight.size(2), weight.size(3)), weight_norm_method)

def load_resnet_imagenet_sketch_model(model):
    cfg = {'resnet18': [2, 2, 2, 2],
           'resnet34': [3, 4, 6, 3],
           'resnet50': [3, 4, 6, 3],
           'resnet101': [3, 4, 23, 3],
           'resnet152': [3, 8, 36, 3]}

    if args.sketch_model is None or not os.path.exists(args.sketch_model):
        raise ('Sketch model path should be exist!')
    ckpt = torch.load(args.sketch_model, map_location=device)
    origin_model = import_module(f'model.{args.arch}_imagenet').resnet(args.cfg).to(device)
    origin_model.load_state_dict(ckpt)
    logger.info('==>Before Sketch')
    test(origin_model, testLoader, topk=(1, 5))

    oristate_dict = origin_model.state_dict()

    state_dict = model.state_dict()
    is_preserve = False  # Whether the previous layer retains the original weight dimension, no sketch

    current_cfg = cfg[args.cfg]

    all_sketch_conv_weight = []
    all_sketch_bn_weight = []

    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for i in range(num):
            if args.cfg == 'resnet18' or args.cfg == 'resnet34':
                iter = 2  # the number of convolution layers in a block, except for shortcut
            else:
                iter = 3
            for j in range(iter):
                # Block the first convolution layer, only sketching the first dimension
                # Block the last convolution layer, only Skitch on the channel dimension
                conv_name = layer_name + str(i) + '.conv' + str(j + 1)
                conv_weight_name = conv_name + '.weight'
                all_sketch_conv_weight.append(conv_weight_name)  # Record the weight of the sketch
                oriweight = oristate_dict[conv_weight_name]
                l = state_dict[conv_weight_name].size(0)

                if l < oriweight.size(1) * oriweight.size(2) * oriweight.size(3) and j != iter - 1:
                    bn_weight_name = layer_name + str(i) + '.bn' + str(j + 1) + '.weight'
                    all_sketch_bn_weight.append(bn_weight_name)

                    sketch_filter = sketch_matrix(oriweight, l, dim=0,
                                                  weight_norm_method=args.weight_norm_method)
                    if is_preserve or j == 0:
                        state_dict[conv_weight_name] = sketch_filter
                    else:
                        l = state_dict[conv_weight_name].size(1)
                        sketch_channel = sketch_matrix(sketch_filter, l, dim=1,
                                                       weight_norm_method=args.weight_norm_method)
                        state_dict[conv_weight_name] = sketch_channel
                    is_preserve = False
                else:
                    if j == iter - 1:  # Block the last volume layer only sketch the channel dimension
                        l = state_dict[conv_weight_name].size(1)
                        sketch_channel = sketch_matrix(oriweight, l, dim=1,
                                                       weight_norm_method=args.weight_norm_method)
                        state_dict[conv_weight_name] = sketch_channel
                    else:
                        state_dict[conv_weight_name] = oriweight
                        is_preserve = True

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_sketch_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.BatchNorm2d):
            bn_weight_name = name + '.weight'
            bn_bias_name = name + '.bias'
            bn_mean_name = name + '.running_mean'
            bn_var_name = name + '.running_var'
            if bn_weight_name not in all_sketch_bn_weight:
                state_dict[bn_weight_name] = oristate_dict[bn_weight_name]
                state_dict[bn_bias_name] = oristate_dict[bn_bias_name]
                state_dict[bn_mean_name] = oristate_dict[bn_mean_name]
                state_dict[bn_var_name] = oristate_dict[bn_var_name]

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)
    logger.info('==>After Sketch')
    test(model, testLoader, topk=(1, 5))

def train(model, optimizer, trainLoader, args, epoch, topk=(1,)):

    model.train()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()
    print_freq = trainLoader._size // args.train_batch_size // 10
    start_time = time.time()
    for batch, batch_data in enumerate(trainLoader):

        inputs = batch_data[0]['data'].to(device)
        targets = batch_data[0]['label'].squeeze().long().to(device)

        adjust_learning_rate(optimizer, epoch, batch, trainLoader._size // args.train_batch_size)

        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets, topk=topk)
        accuracy.update(prec1[0], inputs.size(0))
        top5_accuracy.update(prec1[1], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Top1 {:.2f}%\t'
                'Top5 {:.2f}%\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, trainLoader._size,
                    float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), cost_time
                )
            )
            start_time = current_time
    trainLoader.reset()

def test(model, testLoader, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testLoader):
            inputs = batch_data[0]['data'].to(device)
            targets = batch_data[0]['label'].squeeze().long().to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets, topk=topk)
            accuracy.update(predicted[0], inputs.size(0))
            top5_accuracy.update(predicted[1], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                .format(float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), (current_time - start_time))
        )
    testLoader.reset()
    return accuracy.avg, top5_accuracy.avg

def adjust_learning_rate(optimizer, epoch, step, len_epoch):

    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr * (0.1 ** factor)

    #Warmup
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    start_epoch = 0
    best_top1_acc = 0.0
    best_top5_acc = 0.0

    print('==> Building model..')
    sketch_rate = utils.get_sketch_rate(args.sketch_rate)
    model = import_module(f'model.{args.arch}_imagenet')\
                .resnet(args.cfg, sketch_rate=sketch_rate, start_conv=args.start_conv).to(device)
    load_resnet_imagenet_sketch_model(model)

    print('==>Sketch Done!')
    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.num_epochs):
        train(model, optimizer, trainLoader, args, epoch, topk=(1, 5))

        test_top1_acc, test_top5_acc = test(model, testLoader, topk=(1, 5))

        is_best = best_top5_acc < test_top5_acc
        best_top1_acc = max(best_top1_acc, test_top1_acc)
        best_top5_acc = max(best_top5_acc, test_top5_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_top1_acc': best_top1_acc,
            'best_top5_acc': best_top5_acc,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best Top-1 accuracy: {:.3f} Top-5 accuracy: {:.3f}'.format(float(best_top1_acc), float(best_top5_acc)))

if __name__ == '__main__':
    main()