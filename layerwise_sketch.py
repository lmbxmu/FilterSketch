import torch
import torch.nn as nn
import torch.optim as optim
from importlib import import_module

import utils.common as utils
from data import cifar10, imagenet
from utils.options import args
from model.vgg import Layerwise_SketchVGG
from model.resnet import layerwise_resnet, ResBasicBlock

import os
import time

device = torch.device(f"cuda:{args.gpus[0]}")
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

# Data
print('==> Preparing data..')
if args.data_set == 'cifar10':
    loader = cifar10.Data(args)
elif args.data_set == 'imagenet':
    loader = imagenet.Data(args)

def weight_norm(weight, weight_norm_method=None, filter_norm=False):

    if weight_norm_method == 'max':
        norm_func = lambda x: torch.max(torch.abs(x))
    elif weight_norm_method == 'sum':
        norm_func = lambda x: torch.sum(torch.abs(weight))
    elif weight_norm_method == 'l2':
        norm_func = lambda x: torch.sqrt(torch.sum(x.pow(2)))
    elif weight_norm_method == 'l1':
        norm_func = lambda x: torch.sqrt(torch.sum(torch.abs(x)))
    elif weight_norm_method == 'l2_2':
        norm_func = lambda x: torch.sum(weight.pow(2))
    elif weight_norm_method == '2max':
        norm_func = lambda x: (2 * torch.max(torch.abs(x)))
    else:
        norm_func = lambda x: x

    if filter_norm:
        for i in range(weight.size(0)):
            weight[i] /= norm_func(weight[i])
    else:
        weight /= norm_func(weight)

    return weight

def sketch_matrix(weight, l, dim,
                  bn_weight, bn_bias=None, sketch_bn=False,
                  weight_norm_method=None, filter_norm=False):
    # if l % 2 != 0:
    #     raise ('l should be an even number...')
    A = weight.clone()
    if weight.dim() == 4:  #Convolution layer
        A = A.view(A.size(dim), -1)
        if sketch_bn:
            bn_weight = bn_weight.view(bn_weight.size(0), -1)
            A = torch.cat((A, bn_weight), 1)
            bn_bias = bn_bias.view(bn_bias.size(0), -1)
            A = torch.cat((A, bn_bias), 1)

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
            sigmaSquare = sigmaSquare - torch.eye(l) * theta
            sigmaHat = torch.sqrt(torch.where(sigmaSquare > 0,
                                              sigmaSquare, torch.zeros(sigmaSquare.size())))
            B = sigmaHat.mm(u.t())

            numNonzeroRows = ind
            B[numNonzeroRows, :] = A[i, :]

        numNonzeroRows = numNonzeroRows + 1

    if dim == 0:
        if sketch_bn:
            split_size = weight.size(1) * weight.size(2) * weight.size(3)
            B, bn_para = torch.split(B, split_size, dim=1)
            return weight_norm(B.view(l, weight.size(1), weight.size(2), weight.size(3)), weight_norm_method, filter_norm), \
                   torch.unsqueeze(bn_para[:, 0], 0).view(-1), \
                   torch.unsqueeze(bn_para[:, 1], 0).view(-1),
        else:
            return weight_norm(B.view(l, weight.size(1), weight.size(2), weight.size(3)), weight_norm_method, filter_norm)
    elif dim == 1:
        return weight_norm(B.view(weight.size(0), l, weight.size(2), weight.size(3)), weight_norm_method, filter_norm)


# Training
def train(model, optimizer, trainLoader, args, epoch, sketch_layer):

    model.train()
    losses = utils.AverageMeter()
    accurary = utils.AverageMeter()
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

        prec1 = utils.accuracy(output, targets)
        accurary.update(prec1[0], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Layer{:3d}\t'
                'Epoch[{:3d}] ({:5d}/{:5d}):\t'
                'Loss {:.4f}\t'
                'Accurary {:.2f}%\t\t'
                'Time {:.2f}s'.format(
                    sketch_layer,
                    epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                    float(losses.avg), float(accurary.avg), cost_time
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
    return accuracy.avg

def layerwise_sketch_vgg(orimodel, sketch_model, sketch_layer):

    oristate_dict = orimodel.state_dict()
    state_dict = sketch_model.state_dict()
    sketch_last = False #the last layer sketch whether or not
    current_layer = 0
    for name, module in orimodel.named_modules():
        if isinstance(module, nn.Conv2d):

            oriweight = module.weight.data
            bn_index = int(name.split('.')[1]) + 1  # the index of BN in state_dict
            l = int(oriweight.size(0) * args.sketch_rate)

            if l < oriweight.size(1) * oriweight.size(2) * oriweight.size(3):

                if sketch_last: #the current layer need sketch channel
                    l = int(oriweight.size(1) * args.sketch_rate)
                    sketch_channel = sketch_matrix(oriweight, l, dim=1,
                                                  bn_weight=oristate_dict['features.' + str(bn_index) + '.weight'],
                                                  bn_bias=oristate_dict['features.' + str(bn_index) + '.bias'],
                                                  sketch_bn=False, weight_norm_method=args.weight_norm_method,
                                                  filter_norm=args.filter_norm)
                    state_dict[name + '.weight'] = sketch_channel

                if current_layer == sketch_layer:
                    sketch_filter = sketch_matrix(oriweight, l, dim=0,
                                                  bn_weight=oristate_dict['features.' + str(bn_index) + '.weight'],
                                                  bn_bias=oristate_dict['features.' + str(bn_index) + '.bias'],
                                                  sketch_bn=False, weight_norm_method=args.weight_norm_method,
                                                  filter_norm=args.filter_norm)
                    state_dict[name + '.weight'] = sketch_filter
                    sketch_last = True
                else:
                    if sketch_last:
                        sketch_last = False
                    else:
                        state_dict[name + '.weight'] = oriweight

                    state_dict['features.' + str(bn_index) + '.weight'] = \
                        oristate_dict['features.' + str(bn_index) + '.weight']
                    state_dict['features.' + str(bn_index) + '.bias'] = \
                        oristate_dict['features.' + str(bn_index) + '.bias']
                    state_dict['features.' + str(bn_index) + '.running_mean'] = \
                        oristate_dict['features.' + str(bn_index) + '.running_mean']
                    state_dict['features.' + str(bn_index) + '.running_var'] = \
                        oristate_dict['features.' + str(bn_index) + '.running_var']
                    state_dict['features.' + str(bn_index) + '.num_batches_tracked'] = \
                        oristate_dict['features.' + str(bn_index) + '.num_batches_tracked']


            else:
                state_dict[name + '.weight'] = oriweight

            current_layer += 1

        elif isinstance(module, nn.Linear) and sketch_layer != 12:

            state_dict[name + '.weight'] = module.weight.data
            state_dict[name + '.bias'] = module.bias.data

    sketch_model.load_state_dict(state_dict)

def layerwise_sketch_resnet(orimodel, sketch_model, sketch_layer):

    oristate_dict = orimodel.state_dict()
    state_dict = sketch_model.state_dict()
    sketch_last = False #the last layer sketch whether or not
    current_layer = 0

    for name, module in orimodel.named_modules():

        if isinstance(module, ResBasicBlock):

            if current_layer == sketch_layer:

                oriweight = state_dict

    for name, module in orimodel.named_modules():
        if isinstance(module, nn.Conv2d):

            oriweight = module.weight.data
            bn_index = int(name.split('.')[1]) + 1  # the index of BN in state_dict
            l = int(oriweight.size(0) * args.sketch_rate)

            if l < oriweight.size(1) * oriweight.size(2) * oriweight.size(3):

                if sketch_last: #the current layer need sketch channel
                    l = int(oriweight.size(1) * args.sketch_rate)
                    sketch_channel = sketch_matrix(oriweight, l, dim=1,
                                                  bn_weight=oristate_dict['features.' + str(bn_index) + '.weight'],
                                                  bn_bias=oristate_dict['features.' + str(bn_index) + '.bias'],
                                                  sketch_bn=False, weight_norm_method=args.weight_norm_method,
                                                  filter_norm=args.filter_norm)
                    state_dict[name + '.weight'] = sketch_channel

                if current_layer == sketch_layer:
                    sketch_filter = sketch_matrix(oriweight, l, dim=0,
                                                  bn_weight=oristate_dict['features.' + str(bn_index) + '.weight'],
                                                  bn_bias=oristate_dict['features.' + str(bn_index) + '.bias'],
                                                  sketch_bn=False, weight_norm_method=args.weight_norm_method,
                                                  filter_norm=args.filter_norm)
                    state_dict[name + '.weight'] = sketch_filter
                    sketch_last = True
                else:
                    if sketch_last:
                        sketch_last = False
                    else:
                        state_dict[name + '.weight'] = oriweight

                    state_dict['features.' + str(bn_index) + '.weight'] = \
                        oristate_dict['features.' + str(bn_index) + '.weight']
                    state_dict['features.' + str(bn_index) + '.bias'] = \
                        oristate_dict['features.' + str(bn_index) + '.bias']
                    state_dict['features.' + str(bn_index) + '.running_mean'] = \
                        oristate_dict['features.' + str(bn_index) + '.running_mean']
                    state_dict['features.' + str(bn_index) + '.running_var'] = \
                        oristate_dict['features.' + str(bn_index) + '.running_var']
                    state_dict['features.' + str(bn_index) + '.num_batches_tracked'] = \
                        oristate_dict['features.' + str(bn_index) + '.num_batches_tracked']


            else:
                state_dict[name + '.weight'] = oriweight

            current_layer += 1

        elif isinstance(module, nn.Linear) and sketch_layer != 12:

            state_dict[name + '.weight'] = module.weight.data
            state_dict[name + '.bias'] = module.bias.data

    sketch_model.load_state_dict(state_dict)

def main():

    start_epoch = 0
    best_acc = 0.0
    # Model
    print('==> Building model..')

    ckpt = torch.load(args.sketch_model, map_location=device)
    if args.arch == 'vgg':
        origin_model = import_module(f'model.{args.arch}').VGG().to(device)
        origin_model.load_state_dict(ckpt['state_dict'])
    elif args.arch == 'resnet' and args.data_set == 'cifar10':
        origin_model = import_module(f'model.{args.arch}').VGG().to(device)
        origin_model.load_state_dict(ckpt['state_dict']).resnet(args.cfg).to(device)
    elif args.arch == 'resnet' and args.data_set == 'imagenet':
        origin_model = import_module(f'model.{args.arch}_imagenet').resnet(args.cfg).to(device)
        origin_model.load_state_dict(ckpt)

    total_layers = {
        'vgg16': 13 if args.sketch_lastconv else 12,
        'resnet56': 28,
        'resnet110': 55,
        'resnet18': 5,
        'resnet34': 17,
        'resnet50': 17,
        'resnet101': 34,
        'resnet152': 51,
    }

    for sketch_layer in range(1, total_layers[args.cfg], 1):

        if args.arch == 'vgg':
            sketch_model = Layerwise_SketchVGG(sketch_rate=args.sketch_rate, sketch_layer=sketch_layer).to(device)
            layerwise_sketch_vgg(origin_model, sketch_model, sketch_layer)
        elif args.arch == 'resnet' and args.data_set == 'cifar10':
            sketch_model = layerwise_resnet(args.cfg, sketch_rate=args.sketch_rate, sketch_layer=args.sketch_layer)
            layerwise_sketch_resnet(origin_model, sketch_model, sketch_layer)
        elif args.arch == 'resnet' and args.data_set == 'imagenet':
            pass

        if len(args.gpus) != 1:
            sketch_model = nn.DataParallel(sketch_model, device_ids=args.gpus)

        optimizer = optim.SGD(sketch_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)

        best_acc = 0.0

        logger.info('==> Layer%d' % sketch_layer)
        test(sketch_model, loader.testLoader)

        for epoch in range(0, args.num_epochs):

            train(sketch_model, optimizer, loader.trainLoader, args, epoch, sketch_layer)
            scheduler.step()
            test_acc = test(sketch_model, loader.testLoader)

            is_best = best_acc < test_acc
            best_acc = max(best_acc, test_acc)

            model_state_dict = sketch_model.module.state_dict() \
                    if len(args.gpus) > 1 else sketch_model.state_dict()

            state = {
                'state_dict': model_state_dict,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1
            }
            checkpoint.save_model(state, (sketch_layer - 1) * args.num_epochs + epoch + 1, is_best)
        logger.info('Best accurary: {:.3f}'.format(float(best_acc)))
        with open(os.path.join(args.job_dir, 'checkpoint/model_best.pt'), 'rb') as f:
            ckpt = torch.load(f, map_location=device)
            sketch_model.load_state_dict(ckpt['state_dict'])
            origin_model = sketch_model

    logger.info('Best accurary: {:.3f}'.format(float(best_acc)))

if __name__ == '__main__':
    main()