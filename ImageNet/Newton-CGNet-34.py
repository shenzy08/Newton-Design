import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import shutil
import time
import torchvision
import numpy as np

class add_resblock(nn.Module):
    def __init__(self, input_channels, layers, stride = 1, reduction = 4):
        super().__init__()
        self.input_channels = input_channels
        self.layers = layers
        self.stride = stride
        self.reduction = reduction
        
        self.conv_1 = nn.ParameterList([])
        self.conv_2 = nn.ParameterList([])
        #self.conv_3 = nn.ParameterList([])
        
        ## 1st block: inputchannels(64), else: input channels * 2(512, 1024, 2048)
        self.channels = self.input_channels * 1 if stride == 1 else self.input_channels * 2
        
        ## 1st block: 64, else: 128, 256, 512
        self.bottle_channels = self.channels       
        
        for i in range(layers):
            if i == 0:
                param1 = torch.empty(self.bottle_channels, self.input_channels, 3 ,3)
            else:
                param1 = torch.empty(self.bottle_channels, self.channels, 3 ,3)
                
            param2 = torch.empty(self.bottle_channels, self.bottle_channels, 3, 3)
            #param3 = torch.empty(self.channels, self.bottle_channels, 1, 1)
            
            param1 = nn.init.kaiming_normal_(param1, mode='fan_out', nonlinearity='relu')
            param2 = nn.init.kaiming_normal_(param2, mode='fan_out', nonlinearity='relu')
            #param3 = nn.init.kaiming_normal_(param3, mode='fan_out', nonlinearity='relu')

            self.conv_1.append(nn.Parameter(param1))
            self.conv_2.append(nn.Parameter(param2))
            #self.conv_3.append(nn.Parameter(param3))
            
        self.bn_1 = nn.ModuleList([nn.BatchNorm2d(self.bottle_channels) for i in range(layers)])
        self.bn_2 = nn.ModuleList([nn.BatchNorm2d(self.bottle_channels) for i in range(layers)])
        #self.bn_3 = nn.ModuleList([nn.BatchNorm2d(self.channels) for i in range(layers)])
        
        if self.stride > 1:
            self.transit_param = nn.Parameter(nn.init.kaiming_normal_(
                torch.empty(self.channels, self.input_channels, 1, 1), mode = 'fan_out', nonlinearity = 'relu'))
            self.transit_bn = nn.BatchNorm2d(self.channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        
    def forward(self, x):

        
        prob_output = torch.Tensor([]).cuda()
        
        for i in range(self.layers):
            
            if i == 0 and self.stride > 1:
                res = self.transit_bn(F.conv2d(x, self.transit_param, stride = self.stride, padding = 0))
                current = F.conv2d(x, self.conv_1[i], stride = self.stride, padding = 1) ## 1,self.stride,0
            else:
                res = x
                current = F.conv2d(x, self.conv_1[i], stride = 1, padding = 1)
            ### top layer
            current = self.bn_1[i](current)
            current = self.relu(current)
            ### middle layer
            current = F.conv2d(current, self.conv_2[i], stride = 1, padding = 1)   ## 3,1,1
            current = self.bn_2[i](current)
            
            x = self.relu(current + res)
            
        return x, prob_output


class add_cgblock(nn.Module):
    def __init__(self, input_channels, layers, stride = 1, reduction = 4):
        super().__init__()
        self.input_channels = input_channels
        self.layers = layers
        self.stride = stride
        self.reduction = reduction
        
        self.conv_1 = nn.ParameterList([])
        self.conv_2 = nn.ParameterList([])
        self.conv_3 = nn.ParameterList([])
        self.conv_4 = nn.ParameterList([])
        self.alpha = nn.ParameterList([])
        self.beta = nn.ParameterList([])
        #self.conv_3 = nn.ParameterList([])
        
        ## 1st block: inputchannels(64), else: input channels * 2(512, 1024, 2048)
        self.channels = self.input_channels * 1 if stride == 1 else self.input_channels * 2
        
        ## 1st block: 64, else: 128, 256, 512
        self.bottle_channels = self.channels       
        
        for i in range(layers):
            if i == 0:
                param1 = torch.empty(int(self.bottle_channels/4), self.input_channels, 3 ,3)
                param2 = torch.empty(int(self.bottle_channels/4), self.input_channels, 3 ,3)
                param3 = torch.empty(self.bottle_channels, self.input_channels, 3 ,3)
            else:
                param1 = torch.empty(int(self.bottle_channels/4), self.channels, 3 ,3)
                param2 = torch.empty(int(self.bottle_channels/4), self.channels, 3 ,3)
                param3 = torch.empty(self.bottle_channels, self.channels, 3 ,3)
                
            param4 = torch.empty(int(self.bottle_channels/2), self.bottle_channels, 3, 3)
            #param3 = torch.empty(self.channels, self.bottle_channels, 1, 1)
            
            param1 = nn.init.kaiming_normal_(param1, mode='fan_out', nonlinearity='relu')
            param2 = nn.init.kaiming_normal_(param2, mode='fan_out', nonlinearity='relu')
            param3 = nn.init.kaiming_normal_(param3, mode='fan_out', nonlinearity='relu')
            param4 = nn.init.kaiming_normal_(param4, mode='fan_out', nonlinearity='relu')
            alpha = torch.ones(1) * 0.1
            beta = torch.ones(1) * 0.1

            self.conv_1.append(nn.Parameter(param1))
            self.conv_2.append(nn.Parameter(param2))
            self.conv_3.append(nn.Parameter(param3))
            self.conv_4.append(nn.Parameter(param4))
            self.alpha.append(nn.Parameter(alpha))
            self.beta.append(nn.Parameter(beta))

            
        self.bn_1 = nn.ModuleList([nn.BatchNorm2d(int(self.bottle_channels/4)) for i in range(layers)])
        self.bn_2 = nn.ModuleList([nn.BatchNorm2d(int(self.bottle_channels/4)) for i in range(layers)])
        self.bn_3 = nn.ModuleList([nn.BatchNorm2d(self.bottle_channels) for i in range(layers)])
        self.bn_4 = nn.ModuleList([nn.BatchNorm2d(int(self.bottle_channels/2)) for i in range(layers)])
        
        if self.stride > 1:
            self.transit_param1 = nn.Parameter(nn.init.kaiming_normal_(
                torch.empty(self.channels, self.input_channels, 1, 1), mode = 'fan_out', nonlinearity = 'relu'))
            self.transit_param2 = nn.Parameter(nn.init.kaiming_normal_(
                torch.empty(self.channels, self.input_channels, 1, 1), mode = 'fan_out', nonlinearity = 'relu'))
            self.transit_param3 = nn.Parameter(nn.init.kaiming_normal_(
                torch.empty(self.channels, self.input_channels, 1, 1), mode = 'fan_out', nonlinearity = 'relu'))  

            self.transit_bn1 = nn.BatchNorm2d(self.channels)
            self.transit_bn2 = nn.BatchNorm2d(self.channels)
            self.transit_bn3 = nn.BatchNorm2d(self.channels)

        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(p=0.1)
        
        
    def forward(self, x):

        prob_output = torch.Tensor([]).cuda()

        if self.stride > 1:
            res1 = self.transit_bn1(F.conv2d(x, self.transit_param1, stride = self.stride, padding = 0))           
        else:
            res1 = x

        for i in range(self.layers):
            if i == 0:
                if self.stride > 1:

                    y = x
                    res2 = self.transit_bn2(F.conv2d(y, self.transit_param2, stride = self.stride, padding = 0))
                    res3 = self.transit_bn3(F.conv2d(y, self.transit_param2, stride = self.stride, padding = 0))
                    cg1 = F.conv2d(y, self.conv_1[i], stride = self.stride, padding = 1) ## 1,self.stride,0
                    cg2 = self.dropout(y)
                    cg2 = F.conv2d(cg2, self.conv_2[i], stride = self.stride, padding = 1)
                    cg3 = F.conv2d(y, self.conv_3[i], stride = self.stride, padding = 1)
                else:
                    # res1 = x
                    y = x
                    res2 = x
                    res3 = x
                    cg1 = F.conv2d(y, self.conv_1[i], stride = 1, padding = 1) ## 1,self.stride,0
                    cg2 = self.dropout(y)
                    cg2 = F.conv2d(cg2, self.conv_2[i], stride = 1, padding = 1)
                    cg3 = F.conv2d(y, self.conv_3[i], stride = 1, padding = 1)                                   

                cg1 = self.bn_1[i](cg1)
                cg1 = self.relu(cg1)
                cg1 = self.dropout(cg1)

                cg2 = self.bn_2[i](cg2)
                cg2 = self.relu(cg2)
                
                cg3 = self.bn_3[i](cg3)
                cg3 = self.relu(cg3)
                cg3 = self.dropout(cg3)
                cg3 = F.conv2d(cg3,self.conv_4[i],stride = 1,padding = 1)
                cg3 = self.bn_4[i](cg3)
                cg3 = self.relu(cg3)

                ### middle layer
                g = torch.cat((cg1,cg2,cg3),1) + res3
                d = -g
                y = res2 + self.alpha[i] * d

            else:
                cg1 = F.conv2d(y, self.conv_1[i], stride = 1, padding = 1) ## 1,self.stride,0
                cg1 = self.bn_1[i](cg1)
                cg1 = self.relu(cg1)
                cg1 = self.dropout(cg1)
                
                cg2 = self.dropout(y)
                cg2 = F.conv2d(cg2, self.conv_2[i], stride = 1, padding = 1)
                cg2 = self.bn_2[i](cg2)
                cg2 = self.relu(cg2)
                
                cg3 = F.conv2d(y, self.conv_3[i], stride = 1, padding = 1)            ### top layer
                cg3 = self.bn_3[i](cg3)
                cg3 = self.relu(cg3)
                cg3 = self.dropout(cg3)
                cg3 = F.conv2d(cg3,self.conv_4[i],stride = 1,padding = 1)
                cg3 = self.bn_4[i](cg3)
                cg3 = self.relu(cg3)

                ### middle layer
                g = torch.cat((cg1,cg2,cg3),1) + y
                d = -g + self.beta[i] * d
                y = y + self.alpha[i] * d

            
        x = self.relu(res1 + y)
            
        return x, prob_output


class CgNet_Imagenet(nn.Module):
    def __init__(self, layer_list, num_classes = 1000):
        super().__init__()
        self.first_layer = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
                )
        
        self.block1 = add_resblock(64, layer_list[0])
        self.block2 = add_resblock(64, layer_list[1], stride = 2)
        # self.block3 = add_resblock(128, 1, stride = 2)
        self.block3 = add_cgblock( 128, layer_list[2], stride = 2)
        self.block4 = add_resblock(256, layer_list[3], stride = 2)
        
        self.global_pool = nn.AvgPool2d(7, stride = 1)
        
        self.fc = nn.Linear(512, 1000)
        
        ## initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

    def forward(self, x):
        x = self.first_layer(x)
        
        x, prob_output_1 = self.block1(x)
        x, prob_output_2 = self.block2(x)
        x, prob_output_3 = self.block3(x)
        x, prob_output_4 = self.block4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        #prob_loss = (prob_loss_1 + prob_loss_2 + prob_loss_3 + prob_loss_4) / 4
        prob_output = torch.cat((prob_output_1, prob_output_2, prob_output_3, prob_output_4), 0)
        #print('block1:',prob_output_1.size())
        #print('block2:',prob_output_2.size())
        #print('block3:',prob_output_3.size())
        #print('block4:',prob_output_4.size())
        return x

##   training and testing ##
    

parser = argparse.ArgumentParser(description='ImageNet Training')

#parser.add_argument('data', metavar='DIR',
#                    help='path to the imagenet dataset')
parser.add_argument('--data-path', default='../imagenet/', help='dataset')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 160)')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')

parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--depth', default = '34', help = '50 101 or 152')



best_prec1 = 0


def main():

    global args, best_prec1
    args = parser.parse_args()


    # create model
    depth = args.depth
    layer_info = {'18': [2, 2, 2, 2],
                  '34': [3, 4, 6, 3]}
    layer_list = layer_info[depth]
    model = CgNet_Imagenet(layer_list)
    
    LSTMparam = []       
    
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    
    # print('LSTM params number: %d' % len(LSTMparam))
    
    filter_list = list(map(id, LSTMparam))
    
    other_param = filter(lambda p: id(p) not in filter_list, model.parameters())    
    

    optimizer = torch.optim.SGD([
            {'params': other_param, 'lr': 0.1},
            {'params': LSTMparam, 'lr': 0.1},
            ], momentum = args.momentum, weight_decay = args.weight_decay)


   # # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            print('best_prec1:',best_prec1)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if os.path.exists('top1_acc.npy'):
        record_top1_acc = np.load('top1_acc.npy')
    else:
        record_top1_acc = np.zeros(args.epochs)

    cudnn.benchmark = True
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')

    scale = (0.08, 1.0)
    
    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    get_number_of_param(model)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        record_top1_acc[epoch] = prec1
        np.save('top1_acc.npy',record_top1_acc)
        
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)



def train(train_loader, model, criterion, optimizer, epoch):
    """train model"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model_losses = AverageMeter()
    #struc_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        # pdb.set_trace()
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)

        model_loss = criterion(output, target_var)

        loss = model_loss #+ structural_loss * 0.1
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        
        model_losses.update(model_loss.data.item(), input.size(0))
        #struc_losses.update(structural_loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'model Loss {model_losses.val:.4f} ({model_losses.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, model_losses=model_losses, top1=top1, top5=top5))

    print(time.ctime())


def validate(val_loader, model, criterion):
    """validate model"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    #logit = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()


    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            # compute output
            output = model(input_var)
            #structural_loss = torch.mean(structural_loss)
            #mean_logit = torch.mean(structural_output)
            model_loss = criterion(output, target_var)
            loss = model_loss #+ structural_loss * 0.1
             # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            #logit.update(mean_logit.data.item(), input.size(0)) 
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'model Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg
    

def get_number_of_param(model):
    """get the number of param for every element"""
    count = 0
    for param in model.parameters():
        param_size = param.size()
        count_of_one_param = 1
        for dis in param_size:
            count_of_one_param *= dis
        print(param.size(), count_of_one_param)
        print(count)
        count += count_of_one_param
    print('total number of the model is %d'%count)
    
    
def adjust_learning_rate(optimizer, epoch):
    if epoch == 30 or epoch == 60 or epoch == 90:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
                       
    print('current learning rate for others: %f' % optimizer.param_groups[0]['lr'])
    print('current learning rate for LSTMs: %f' % optimizer.param_groups[1]['lr'])
    
    
    
    
def save_checkpoint(state, is_best, filename='./model/Cg34_baseline.pth'):
    """Save the trained model"""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './model/Cg34_baseline_best.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim = True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res

if __name__ == '__main__':
    main()









    
    
    
    
