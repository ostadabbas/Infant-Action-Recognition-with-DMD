import torch
import glob
import os
import os.path as osp

def save_weights(model, name, work_dir, rem_prev=None):
    if rem_prev is not None:
        pre_results = glob.glob(osp.join(work_dir, f"{rem_prev}*.pth"))
        assert len(pre_results)<=1
        if len(pre_results)==1:
            os.remove(pre_results[0])
    torch.save(model.state_dict(), osp.join(work_dir, name))