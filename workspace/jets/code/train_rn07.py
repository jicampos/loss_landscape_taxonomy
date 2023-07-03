from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.insert(1, './code/')

from model import get_new_model

from arguments import get_parser
from utils import *
from data import get_loader

from sklearn.metrics import accuracy_score

parser = get_parser(code_type='training')


# class from https://github.com/Zhen-Dong/HAWQ/blob/main/quant_train.py#L683
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


# class from https://github.com/Zhen-Dong/HAWQ/blob/main/quant_train.py#L708
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

def save_early_stop_model(args, model, loss_vals):
    
    if not args.save_early_stop or len(loss_vals)<30:
        
        print("Early stopping not satisfied.")
        
        return False
    
    else:
        
        for i in range(args.patience):
            
            if (loss_vals[-1-i] < loss_vals[-2-i] - args.min_delta) or (loss_vals[-1-i] > loss_vals[-2-i] + args.min_delta):
                
                print("Early stopping not satisfied.")
                
                return False
        
        args.save_early_stop = False
        
        print("Early stopping satisfied!!! Saving early stopped model.")
        
        return True


def train(args, model, train_loader, test_loader, optimizer, criterion, epoch):
    model.train()
    
    losses = AverageMeter("Loss", ":.4e")
    accuracy = AverageMeter("Acc", ":6.6f")
    progress = ProgressMeter(
        len(train_loader),
        [losses, accuracy],
        prefix="Epoch: [{}]".format(epoch),
    )
    
    #print('Starting Epoch: ', epoch)
    train_loss = 0.
    total_num = 0
    correct = 0
    
    P = 0 # num samples / batch size
    for i, (inputs, targets) in enumerate(train_loader):
        #print(inputs.shape)
        if args.ignore_incomplete_batch:
            if_condition = inputs.shape[0] != args.train_bs
            
            if if_condition:
                print("Neglect the last batch so that num samples/batch size = int")
                break
                  
        P += 1
        # loop over dataset
        # inputs, targets = inputs.to("cuda"), targets.to("cuda")
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        #print(inputs[0])
        #print(targets[0])
        #print(outputs[0])
        
        #train_loss += loss.item() * targets.size()[0]
        #total_num += targets.size()[0]
        #_, predicted = outputs.max(1)
        #correct += predicted.eq(targets).sum().item()
        
        losses.update(loss.item(), inputs.size(0))

        batch_preds = torch.max(outputs, 1)[1]
        batch_acc = accuracy_score(
            targets.detach().cpu().numpy(), batch_preds.detach().cpu().numpy()
        )
        # update progress meter
        accuracy.update(batch_acc, inputs.size(0))
        
        if i % 300 == 0: # print every 50 batches
            progress.display(i)
            #print(targets[0])
            #print(outputs[0])
            #print(batch_preds)
            #print(batch_labels)
        
        
        loss.backward()
        optimizer.step()
            
    acc, loss = test(args, model, test_loader, criterion)
    train_loss = losses.avg
    print(f"Training Loss of Epoch {epoch}: {losses.avg}")
    print(f"Training Acc of Epoch {epoch}: {accuracy.avg}")
    print(f"Testing Acc of Epoch {epoch}: {acc}")  
    print(f"Testing Loss of Epoch {epoch}: {loss}")  
    
    return train_loss


def main():
    
    args = parser.parse_args()
    
    if args.save_final or args.no_lr_decay or args.one_lr_decay:
        if args.saving_folder == '':
            raise ('you must give a position and name to save your model')
        if args.saving_folder[-1] != '/':
            args.saving_folder += '/'

    for arg in vars(args):
        print(arg, getattr(args, arg))

    print("------------------------------------------------------")
    print("Experiement: {0} training for {1}".format(args.training_type, args.arch))
    print("------------------------------------------------------")

    criterion = nn.CrossEntropyLoss()
    model = get_new_model(args)
    
    #from torchsummary import summary
    #summary(model, input_size=(1, 16))
    print("---------------------- Model -------------------------")
    print(model)
    print("------------------------------------------------------")
    
    if args.resume:
        
        model.load_state_dict(torch.load(f"{args.resume}"))
    
    train_loader, test_loader = get_loader(args)

    if args.training_type == 'small_lr':
        base_lr = 0.003
    else:
        base_lr = args.lr

    print("The base learning rate is {0}".format(base_lr))

    if args.training_type == 'small_lr':
        optimizer = optim.Adam(model.parameters(), lr=base_lr,  weight_decay=args.weight_decay)
    elif args.training_type == 'no_decay':
        optimizer = optim.Adam(model.parameters(), lr=base_lr,  weight_decay=0)
    else:
        optimizer = optim.Adam(model.parameters(), lr=base_lr,  weight_decay=0)
        # optimizer = optim.Adam(model.parameters(), lr=base_lr,  weight_decay=args.weight_decay)

    loss_vals = []
    best_train_loss = 1000000
    
    for epoch in range(args.epochs):

        print("---------------------")
        print("Start epoch {0}".format(epoch))
        print("---------------------")
            
        if args.no_lr_decay:
            lr = base_lr
        elif args.one_lr_decay:
            # Here, the training is done with one learning rate decay
            # So it's hard to justify what temperature is used
            # Therefore, we train with longer first period
            if epoch >= args.epochs*0.75:
                lr = base_lr * 0.1
            else:
                lr = base_lr
        
        update_lr(optimizer, lr)        

        train_loss = train(args, model, train_loader, test_loader, optimizer, criterion, epoch)
        
        loss_vals.append(train_loss)
        
        if args.save_best and train_loss < best_train_loss:
            print('Model with the best training loss saved! The loss is {0}'.format(train_loss))
            torch.save(model.state_dict(), f'{args.saving_folder}net_{args.file_prefix}_best.pkl')
            best_train_loss = train_loss
        
        if args.only_exploration and epoch >= args.epochs*0.5:
            print("only log the process before lr decay")
            break
        
        if save_early_stop_model(args, model, loss_vals):
            torch.save(model.state_dict(), f'{args.saving_folder}net_{args.file_prefix}_early_stopped_model.pkl')
            
        if args.no_lr_decay and epoch==args.stop_epoch:
            torch.save(model.state_dict(), f'{args.saving_folder}net_{args.file_prefix}.pkl')
            print("Early stopping without learning rate decay.")
            break
            
        if args.one_lr_decay and epoch==args.stop_epoch:
            torch.save(model.state_dict(), f'{args.saving_folder}net_{args.file_prefix}.pkl')
            print("Early stopping with only one lr decay")
            break
        
        if (epoch%5 ==0 or epoch == args.epochs-1) and args.save_final:
            torch.save(model.state_dict(), f'{args.saving_folder}net_{args.file_prefix}.pkl')
            
        if args.save_middle and (epoch%args.save_frequency ==0 or epoch == args.epochs-1):
            torch.save(model.state_dict(), f'{args.saving_folder}net_{epoch}.pkl')

if __name__ == '__main__':
    main()


# For debugging only 
# python code/train_rn07.py --training-type lr_decay --arch RN07_32b --saving-folder ../checkpoint/different_knobs_subset_10/lr_0.1/lr_decay/RN07_32b/ --file-prefix exp_0 --mixup-alpha 16.0 --data-subset --subset 1.0 --data-path ../../data/RN07 --lr 0.1 --weight-decay 0.0005 --train-bs 1024 --test-bs 1024 --weight-precision 32 --bias-precision 32 --act-precision 35  --one-lr-decay --epochs 100 --save-best --ignore-incomplete-batch > >(tee -a ../checkpoint/different_knobs_subset_10/lr_0.1/lr_decay/RN07_32b/log_0.txt) 2> >(tee -a ../checkpoint/different_knobs_subset_10/lr_0.1/lr_decay/RN07_32b/err_0.txt >&2) --experiment-model RN07
# python code/train_rn07.py --training-type lr_decay --arch RN07_6b --saving-folder ../checkpoint/different_knobs_subset_10/lr_0.1/lr_decay/RN07_6b/ --file-prefix exp_0 --mixup-alpha 16.0 --data-subset --subset 1.0 --data-path ../../data/RN07 --lr 0.1 --weight-decay 0.0005 --train-bs 1024 --test-bs 1024 --weight-precision 6 --bias-precision 6 --act-precision 9  --one-lr-decay --epochs 100 --save-best --ignore-incomplete-batch > >(tee -a ../checkpoint/different_knobs_subset_10/lr_0.1/lr_decay/RN07_6b/log_0.txt) 2> >(tee -a ../checkpoint/different_knobs_subset_10/lr_0.1/lr_decay/RN07_6b/err_0.txt >&2) 
