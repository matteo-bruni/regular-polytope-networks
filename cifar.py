'''
Train CIFAR10 with PyTorch.
extended from https://github.com/kuangliu/pytorch-cifar

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as sched

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from prototypes import dsimplex, dcube, polygonal2d, hadamard, orthoplex
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, help='')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='')

parser.add_argument('--batch-size', default=128, type=int, dest='batch_size', help='batch size (default: 128)')
parser.add_argument('--schedule', default='150,250,350', type=str, help='lr schedule + max epochs (default: 150,250,350)')
parser.add_argument('--schedule-gamma', default=0.1, type=float, help='gamma for scheduler')

parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--parallel', action='store_true')

available_nets = ['resnet50', 'senet18', 'densenet169']
parser.add_argument('--net', default='resnet50', type=str, choices=available_nets, required=True)

available_prototypes = ['simplex', 'dcube', 'polygonal2d', 'hadamard', 'orthoplex', 'trainable']
parser.add_argument('--prototype', choices=available_prototypes,
                    required=True, help='choose one of the available weight directions: {}'.format(available_prototypes))
parser.add_argument('--trainable-dim', default=2048, type=int, help='')

available_datasets = ['cifar10', 'cifar100']
parser.add_argument('--dataset', default='cifar10', type=str, choices=available_datasets)
args = parser.parse_args()


*args.schedule, args.max_epochs = map(int, args.schedule.split(','))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0        # best test accuracy
start_epoch = 0     # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

if args.dataset == 'cifar10':

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    n_classes = 10
    feat_dim = None

elif args.dataset == 'cifar100':

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    n_classes = 100
    feat_dim = None

else:
    raise ValueError(f'Unknown dataset, {args.dataset}')


# Model
print('==> Building model..')
if args.net == 'resnet50':
    from models.cifar.resnet import ResNet50
    net_type = ResNet50

elif args.net == 'senet18':
    from models.cifar.resnet import SENet18
    net_type = SENet18

elif args.net == 'densenet169':
    from models.cifar.resnet import DenseNet169
    net_type = DenseNet169

else:
    raise ValueError('network model not supported')


if args.prototype == 'simplex':

    if args.dataset == 'cifar10':
        feat_dim = 9
    elif args.dataset == 'cifar100':
        feat_dim = 99
    else:
        raise ValueError(f'Unknown dataset, {args.dataset}')

    print(f"using {args.prototype}, feat_dim: {feat_dim}, n_classes {n_classes}")
    net = net_type(feat_dim=feat_dim, num_classes=n_classes)
    # assign fixed values
    fixed_weights = torch.from_numpy(dsimplex(num_classes=n_classes))

    # print for debug purposes
    print("fixed_weights", fixed_weights.size())
    print("net.fc.weight", net.fc.weight.size())
    print("net.linear", net.linear)

    net.linear.weight.requires_grad = False     # set no gradient for the fixed classifier
    net.linear.weight.copy_(fixed_weights)      # set the weights for the classifier

elif args.prototype == 'dcube':

    if args.dataset == 'cifar10':
        # IN CIFAR10 feat dim is 9 and num_classes is 16
        feat_dim = 4
        n_classes = 16

    elif args.dataset == 'cifar100':
        # IN CIFAR100 feat dim is 9 and num_classes is 16
        feat_dim = 7
        n_classes = 128
    else:
        raise ValueError(f'Unknown dataset, {args.dataset}')


    print(f"using {args.prototype}, feat_dim: {feat_dim}, n_classes {n_classes}")
    net = net_type(feat_dim=feat_dim, num_classes=n_classes)
    # assign fixed values
    fixed_weights = torch.from_numpy(dcube(num_classes=n_classes, feat_dim=feat_dim))

    print("fixed_weights", fixed_weights.size())
    print("net.fc.weight", net.fc.weight.size())

    net.linear.weight.requires_grad = False  # set no gradient for the fixed classifier
    net.linear.weight.copy_(fixed_weights)  # set the weights for the classifier

elif args.prototype == 'orthoplex':

    if args.dataset == 'cifar10':
        # IN CIFAR10 feat dim is 5
        feat_dim = 5
    elif args.dataset == 'cifar100':
        feat_dim = 50
    else:
        raise ValueError(f'Unknown dataset, {args.dataset}')

    print(f"using {args.prototype}, feat_dim: {feat_dim}, n_classes {n_classes}")
    net = net_type(feat_dim=feat_dim, num_classes=n_classes)
    # assign fixed values
    fixed_weights = torch.from_numpy(orthoplex(num_classes=n_classes, feat_dim=feat_dim))
    net.linear.weight.requires_grad = False  # set no gradient for the fixed classifier
    net.linear.weight.copy_(fixed_weights)  # set the weights for the classifier

elif args.prototype == 'polygonal2d':

    feat_dim = 2
    print(f"using {args.prototype}, feat_dim: {feat_dim}, n_classes {n_classes}")
    net = net_type(feat_dim=feat_dim, num_classes=n_classes)
    # assign fixed values
    fixed_weights = torch.from_numpy(polygonal2d(num_classes=n_classes))
    net.linear.weight.requires_grad = False     # set no gradient for the fixed classifier
    net.linear.weight.copy_(fixed_weights)      # set the weights for the classifier

elif args.prototype == 'hadamard':

    # IN CIFAR10 feat dim is provided
    feat_dim = args.trainable_dim
    print(f"using {args.prototype}, feat_dim: {feat_dim}, n_classes {n_classes}")
    net = net_type(feat_dim=feat_dim, num_classes=n_classes)
    # assign fixed values
    fixed_weights = torch.from_numpy(hadamard(num_classes=n_classes, feat_dim=feat_dim))
    net.linear.weight.requires_grad = False     # set no gradient for the fixed classifier
    net.linear.weight.copy_(fixed_weights)      # set the weights for the classifier

elif args.prototype == 'trainable':
    feat_dim = args.trainable_dim
    print(args.trainable_dim)
    print(f"using {args.prototype}, feat_dim: {feat_dim}, n_classes {n_classes}")
    net = net_type(feat_dim=feat_dim, num_classes=n_classes)

else:
    raise ValueError(f'Unknown prototypes {args.prototype}')

net = net.to(device)

if device == 'cuda' and args.parallel:
    print("Using DataParallel")
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print(str(net))


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)

if len(args.schedule) == 0:
    print('NOT using scheduler')
    scheduler = None
else:
    print('Using scheduler with schedule:', args.schedule)
    scheduler = sched.MultiStepLR(optimizer,
                                  milestones=args.schedule,
                                  gamma=args.schedule_gamma)


# Training


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

    return best_acc


for epoch in range(start_epoch, args.max_epochs):
    train(epoch)
    test(epoch)
    if scheduler is not None:
        scheduler.step()
    print(f"Best Acc: {best_acc}")

