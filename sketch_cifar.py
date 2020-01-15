import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args
from model.googlenet import Inception
import utils.common as utils

import os
import time
from data import cifar10
from importlib import import_module

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

# Data
print('==> Preparing data..')
loader = cifar10.Data(args)

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

def load_resnet_sketch_model(model):
    cfg = {'resnet56': [9, 9, 9],
           'resnet110': [18, 18, 18],
           }

    if args.sketch_model is None or not os.path.exists(args.sketch_model):
        raise ('Sketch model path should be exist!')
    ckpt = torch.load(args.sketch_model, map_location=device)
    origin_model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    origin_model.load_state_dict(ckpt['state_dict'])
    logger.info('==>Before Sketch')
    test(origin_model, loader.testLoader)

    oristate_dict = origin_model.state_dict()

    state_dict = model.state_dict()
    is_preserve = False #Whether the previous layer retains the original weight dimension, no sketch

    current_cfg = cfg[args.cfg]

    all_sketch_conv_weight = []
    all_sketch_bn_weight = []

    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for i in range(num):
            for j in range(2):
                #Block the first convolution layer, only sketching the first dimension
                #Block the last convolution layer, only Skitch on the channel dimension
                conv_name = layer_name + str(i) + '.conv' + str(j + 1)
                conv_weight_name = conv_name + '.weight'
                all_sketch_conv_weight.append(conv_weight_name) #Record the weight of the sketch
                oriweight = oristate_dict[conv_weight_name]
                l = state_dict[conv_weight_name].size(0)

                if l < oriweight.size(1) * oriweight.size(2) * oriweight.size(3) and j == 0:
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
                    if j == 1: #Block the last volume layer only sketch the channel dimension
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
    test(model, loader.testLoader)

def load_googlenet_sketch_model(model):
    if args.sketch_model is None or not os.path.exists(args.sketch_model):
        raise ('Sketch model path should be exist!')
    ckpt = torch.load(args.sketch_model, map_location=device)
    origin_model = import_module(f'model.{args.arch}').googlenet().to(device)
    origin_model.load_state_dict(ckpt['state_dict'])
    logger.info('==>Before Sketch')
    test(origin_model, loader.testLoader)
    oristate_dict = origin_model.state_dict()

    state_dict = model.state_dict()
    all_sketch_conv_name = []
    all_sketch_bn_name = []

    for name, module in origin_model.named_modules():

        if isinstance(module, Inception):

            sketch_filter_channel_index = ['.branch5x5.3']  # the index of sketch filter and channel weight
            sketch_channel_index = ['.branch3x3.3', '.branch5x5.6']  # the index of sketch channel weight
            sketch_filter_index = ['.branch3x3.0', '.branch5x5.0']  # the index of sketch filter weight
            sketch_bn_index = ['.branch3x3.1', '.branch5x5.1', '.branch5x5.4'] #the index of sketch bn weight

            for bn_index in sketch_bn_index:
                all_sketch_bn_name.append(name + bn_index)

            for weight_index in sketch_filter_channel_index:

                conv_name = name + weight_index + '.weight'
                all_sketch_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                l = state_dict[conv_name].size(0)

                sketch_filter = sketch_matrix(oriweight, l, dim=0,
                                              weight_norm_method=args.weight_norm_method)
                l = state_dict[conv_name].size(1)
                sketch_channel = sketch_matrix(sketch_filter, l, dim=1,
                                               weight_norm_method=args.weight_norm_method)
                state_dict[conv_name] = sketch_channel

            for weight_index in sketch_channel_index:

                conv_name = name + weight_index + '.weight'
                all_sketch_conv_name.append(name + weight_index)
                oriweight = oristate_dict[conv_name]

                l = state_dict[conv_name].size(1)
                sketch_channel = sketch_matrix(oriweight, l, dim=1,
                                               weight_norm_method=args.weight_norm_method)
                state_dict[conv_name] = sketch_channel

            for weight_index in sketch_filter_index:

                conv_name = name + weight_index + '.weight'
                all_sketch_conv_name.append(name + weight_index)
                oriweight = oristate_dict[conv_name]

                l = state_dict[conv_name].size(0)
                sketch_filter = sketch_matrix(oriweight, l, dim=0,
                                               weight_norm_method=args.weight_norm_method)
                state_dict[conv_name] = sketch_filter

    for name, module in model.named_modules(): #Reassign non sketch weights to the new network

        if isinstance(module, nn.Conv2d):

            if name not in all_sketch_conv_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']

        elif isinstance(module, nn.BatchNorm2d):

            if name not in all_sketch_bn_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']
                state_dict[name + '.running_mean'] = oristate_dict[name + '.running_mean']
                state_dict[name + '.running_var'] = oristate_dict[name + '.running_var']

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)
    logger.info('==>After Sketch')
    test(model, loader.testLoader)

def train(model, optimizer, trainLoader, args, epoch, topk=(1,)):

    model.train()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets, topk=topk)
        accuracy.update(prec1[0], inputs.size(0))
        if len(topk) == 2:
            top5_accuracy.update(prec1[1], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            if len(topk) == 1:
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'Loss {:.4f}\t'
                    'Accuracy {:.2f}%\t\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                        float(losses.avg), float(accuracy.avg), cost_time
                    )
                )
            else:
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'Loss {:.4f}\t'
                    'Top1 {:.2f}%\t'
                    'Top5 {:.2f}%\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                        float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), cost_time
                    )
                )
            start_time = current_time

def test(model, testLoader, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets, topk=topk)
            accuracy.update(predicted[0], inputs.size(0))
            if len(topk) == 2:
                top5_accuracy.update(predicted[1], inputs.size(0))

        current_time = time.time()
        if len(topk) == 1:
            logger.info(
                'Test Loss {:.4f}\tAccuracy {:.2f}%\t\tTime {:.2f}s\n'
                .format(float(losses.avg), float(accuracy.avg), (current_time - start_time))
            )
        else:
            logger.info(
                'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                    .format(float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), (current_time - start_time))
            )
    if len(topk) == 1:
        return accuracy.avg
    else:
        return top5_accuracy.avg

def main():
    start_epoch = 0
    best_acc = 0.0

    # Model
    print('==> Building model..')
    sketch_rate = utils.get_sketch_rate(args.sketch_rate)
    if args.arch == 'resnet':
        model = import_module(f'model.{args.arch}')\
                        .resnet(args.cfg, sketch_rate=sketch_rate, start_conv=args.start_conv).to(device)
        load_resnet_sketch_model(model)
    elif args.arch == 'googlenet':
        model = import_module(f'model.{args.arch}').googlenet(sketch_rate).to(device)
        load_googlenet_sketch_model(model)
    else:
        raise('arch not exist!')
    print('==>Sketch Done!')

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)

    for epoch in range(start_epoch, args.num_epochs):
        train(model, optimizer, loader.trainLoader, args, epoch, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        scheduler.step()
        test_acc = test(model, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))

        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best accuracy: {:.3f}'.format(float(best_acc)))

if __name__ == '__main__':
    main()