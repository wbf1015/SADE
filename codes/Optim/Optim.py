import torch
from torch import optim

def getoptimizer(args, params):
    
    assert args.optimizer in ['SGD', 'Adam', 'NAdam', 'Adam_rec', 'Adagrad']
    
    if args.optimizer == 'SGD':
        return torch.optim.SGD(params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    
    elif args.optimizer == 'Adam':
        return torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    elif args.optimizer == 'Adam_rec':
        return torch.optim.Adam(params, lr=args.learning_rate, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    
    elif args.optimizer == 'NAdam':
        return torch.optim.NAdam(params, lr=args.learning_rate,  weight_decay=args.weight_decay)
    
    elif args.optimizer == 'Adagrad':
        return torch.optim.Adagrad(params, lr=args.learning_rate,  weight_decay=args.weight_decay)


def getscheduler(args, optimizer, last_epoch=-1):
    
    assert args.scheduler in ['MultiStepLR', 'StepLR', 'Cosine', 'Plateau']

    if args.scheduler == 'MultiStepLR':
        milestones = []
        start = args.warm_up_steps
        while start<=args.steps:
            milestones.append(start)
            start = start*2
        print('milestones=', milestones)
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.decreasing_lr, last_epoch=last_epoch)
    
    elif args.scheduler == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, args.warm_up_steps, gamma=args.decreasing_lr, last_epoch=last_epoch)
    
    elif args.scheduler == 'Cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.warm_up_steps, eta_min=args.learning_rate*args.decreasing_lr, last_epoch=last_epoch)
    
    elif args.scheduler == 'Plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.decreasing_lr, patience=args.patience, cooldown=args.cooldown)


def get_origin_optimizer(args, params):
    if 'TransE' in args.pruning_stratgy:
        pass
    
    if 'RotatE' in args.pruning_stratgy:
        if 'FB15k-237' in args.data_path:
            return torch.optim.Adam(params, lr=0.00005)
        if 'wn18rr' in args.data_path:
            return torch.optim.Adam(params, lr=0.00005)
    
    if 'ComplexE' in args.pruning_stratgy:
        pass
    
    if 'SimplE' in args.pruning_stratgy:
        pass
    