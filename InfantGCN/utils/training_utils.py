import torch
import glob
import os
import os.path as osp
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, MultiStepLR, CyclicLR, ExponentialLR, OneCycleLR

def save_weights(model, name, work_dir, rem_prev=None):
    if rem_prev is not None:
        pre_results = glob.glob(osp.join(work_dir, f"{rem_prev}*.pth"))
        assert len(pre_results)<=1
        if len(pre_results)==1:
            os.remove(pre_results[0])
    torch.save(model.state_dict(), osp.join(work_dir, name))

class CompoundScheduler:
    def __init__(self, scheds, stps):
        self.scheds = scheds
        self.stps = stps

    def __call__(self, num_epoch):
        for step, scheduler in zip(self.stps, self.scheds):
            if num_epoch < step:
                return scheduler
            
def get_lr_scheduler(optimizer, lr_scheduler, num_epochs, num_iters, scheduler_kwargs={}):
    
    if lr_scheduler == "CosineAnnealingLRwithWarmup":
        steps = [0.4*num_epochs,1*num_epochs]
        scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=int(num_iters * 0.4 * num_epochs))
        scheduler2 = CosineAnnealingLR(optimizer, eta_min=scheduler_kwargs['min_lr'], T_max=int(num_iters * 0.6 * num_epochs))
        schedulers = [scheduler1,scheduler2]
        compound_scheduler = CompoundScheduler(schedulers, steps)

    elif lr_scheduler == "CosineAnnealingLRwithWarmupFixed":    
        steps = [0.2*num_epochs,0.5*num_epochs,1*num_epochs]
        scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=int(num_iters * 0.2 * num_epochs))
        scheduler2 = CosineAnnealingLR(optimizer, eta_min=scheduler_kwargs['min_lr'], T_max=int(num_iters * 0.3 * num_epochs))
        scheduler3 = LinearLR(optimizer, start_factor=1, total_iters=int(num_iters * 0.5 * num_epochs))
        schedulers = [scheduler1,scheduler2,scheduler3]
        compound_scheduler = CompoundScheduler(schedulers, steps)

    elif lr_scheduler == "MultiStepLR":    
        scheduler1 = MultiStepLR(optimizer, milestones=[(1/4)*num_epochs*num_iters,(2/4)*num_epochs*num_iters], gamma=scheduler_kwargs['gamma'])
        schedulers = [scheduler1]
        steps = [1*num_epochs]
        compound_scheduler = CompoundScheduler(schedulers, steps)

    elif lr_scheduler == "CycleLR":    
        scheduler1 = CyclicLR(optimizer, 
                              base_lr=0.1*scheduler_kwargs['base_lr'], max_lr=scheduler_kwargs['base_lr'], 
                              step_size_up=(1/4)*num_iters*(num_epochs/scheduler_kwargs['num_cycles']), 
                              step_size_down=(3/4)*num_iters*(num_epochs/scheduler_kwargs['num_cycles']), 
                              mode='exp_range', gamma=scheduler_kwargs['gamma'], scale_mode='cycle')
        schedulers = [scheduler1]
        steps = [1*num_epochs]
        compound_scheduler = CompoundScheduler(schedulers, steps)

    elif lr_scheduler == "ExponentialLRwithWarmup":
        scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=int(num_iters * 0.2 * num_epochs))
        scheduler2 = ExponentialLR(optimizer, scheduler_kwargs['gamma'])
        schedulers = [scheduler1, scheduler2]
        steps = [0.2*num_epochs,1*num_epochs]
        compound_scheduler = CompoundScheduler(schedulers, steps)

    elif lr_scheduler == "OneCycleLR":
        scheduler1 = OneCycleLR(optimizer, total_steps = int(num_iters*num_epochs),
                              max_lr=scheduler_kwargs['base_lr'], 
                              pct_start = scheduler_kwargs['pct_start'], anneal_strategy='cos')
        schedulers = [scheduler1]
        steps = [1*num_epochs]
        compound_scheduler = CompoundScheduler(schedulers, steps)

    else:
        raise ValueError(f"Invalid lr_scheduler: {lr_scheduler}")

    return compound_scheduler