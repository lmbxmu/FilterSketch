import torch
import torch.nn as nn
from utils.options import args
import utils.common as utils

import time
from data import cifar10, imagenet_dali, imagenet
from importlib import import_module

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
loss_func = nn.CrossEntropyLoss()

# Data
print('==> Preparing data..')
if args.data_set == 'cifar10':
    testLoader = cifar10.Data(args).testLoader
else: #imagenet
    if device != 'cpu':
        testLoader = imagenet_dali.get_imagenet_iter_dali('val', args.data_path, args.eval_batch_size,
                                             num_threads=4, crop=224, device_id=args.gpus[0], num_gpus=1)
    else:
        testLoader = imagenet.Data(args).testLoader

def test(model, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testLoader):
            if len(topk) == 2:
                inputs = batch_data[0]['data'].to(device)
                targets = batch_data[0]['label'].squeeze().long().to(device)
            else:
                inputs = batch_data[0]
                targets = batch_data[1]
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
            print(
                'Test Loss {:.4f}\tAccuracy {:.2f}%\t\tTime {:.2f}s\n'
                .format(float(losses.avg), float(accuracy.avg), (current_time - start_time))
            )
        else:
            print(
                'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                    .format(float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), (current_time - start_time))
            )


def main():

    # Model
    print('==> Building model..')
    sketch_rate = utils.get_sketch_rate(args.sketch_rate)
    if args.arch == 'resnet':
        if args.data_set == 'cifar10':
            model = import_module(f'model.{args.arch}')\
                            .resnet(args.cfg, sketch_rate=sketch_rate, start_conv=args.start_conv).to(device)
        else:
            model = import_module(f'model.{args.arch}_imagenet') \
                .resnet(args.cfg, sketch_rate=sketch_rate, start_conv=args.start_conv).to(device)
    elif args.arch == 'googlenet':
        model = import_module(f'model.{args.arch}').googlenet(sketch_rate).to(device)
    else:
        raise('arch not exist!')
    ckpt = torch.load(args.sketch_model, map_location=device)
    model.load_state_dict(ckpt['state_dict'])

    test(model, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))


if __name__ == '__main__':
    main()
