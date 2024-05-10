import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
import tqdm
import os
import os.path as osp
from datetime import datetime

from torch.utils.data import DataLoader
from torch.optim import SGD

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ChainedScheduler, MultiStepLR

from net.stgcn import STGCN
from net.ctrgcn import CTRGCN
from dataloader import Feeder
from utils.logger import Logger
from utils.training_utils import save_weights
from utils.parsers import get_training_parser
from epoch_runner import EpochRunner
from utils.training_utils import CompoundScheduler


if __name__ == "__main__":
    parser = get_training_parser()
    args = parser.parse_args()

    EPOCHS = args.epochs
    MODEL = args.model
    DATA_PATH = args.data_path
    BASE_LR = args.base_lr
    REPEAT = args.repeat
    now = datetime.now().strftime("%m-%d-%y_%H-%M-%S")
    EXP_NAME = args.exp_name if args.exp_name is not None else f"exp_{now}"
    OUT_FOLDER = args.output_folder if args.output_folder is not None else "../Results_test"
    WORK_DIR = osp.join(OUT_FOLDER, EXP_NAME)
    os.makedirs(WORK_DIR, exist_ok=True)
    
    N_FEATS = 2

    train_dataset = Feeder(DATA_PATH, 'train', window_size=60, random_selection="uniform_choose", repeat=REPEAT, break_samples=True)
    val_dataset = Feeder(DATA_PATH, 'val', window_size=60, random_selection="uniform_choose", break_samples=False)
    

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    iterations_per_epoch = len(train_dataloader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # import ipdb
    # ipdb.set_trace()

    graph_args = {'layout': f'infant{N_FEATS}d', 'strategy': 'spatial'}
    kwargs = {}
    input_dim = N_FEATS
    num_classes = 5
    if MODEL=="STGCN":
        model = STGCN(input_dim, num_classes, graph_args, edge_importance_weighting = True, **kwargs).to(device)
    elif MODEL=="CTRGCN":
        model = CTRGCN(input_dim, num_classes, graph_args, **kwargs).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    ce_loss = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR)#SGD(weight_decay=0.01, lr=0.1, params=stgcn.parameters())
        
    """
    steps = [0.2*EPOCHS,0.5*EPOCHS,1*EPOCHS]
    scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=int(iterations_per_epoch * 0.2 * EPOCHS))
    scheduler2 = CosineAnnealingLR(optimizer, eta_min=0.001, T_max=int(iterations_per_epoch * 0.3 * EPOCHS))
    scheduler3 = LinearLR(optimizer, start_factor=1, total_iters=int(iterations_per_epoch * 0.5 * EPOCHS))
    schedulers = [scheduler1,scheduler2,scheduler3]
    compound_scheduler = CompoundScheduler(schedulers, steps)
    """

    scheduler1 = MultiStepLR(optimizer, milestones=[(1/4)*EPOCHS*iterations_per_epoch,(2/4)*EPOCHS*iterations_per_epoch], gamma=0.1)
    schedulers = [scheduler1]
    steps = [1*EPOCHS]
    compound_scheduler = CompoundScheduler(schedulers, steps)

    train_logger = Logger('train', osp.join(WORK_DIR, 'train_log.txt'))
    train_logger.log_message(f"Training parameters:")
    train_logger.log_message(f"Model: {MODEL}")
    train_logger.log_message(f"Data: {DATA_PATH}")
    train_logger.log_message(f"Base learning rate: {BASE_LR}")
    train_logger.log_message(f"Repeat: {REPEAT}")
    train_logger.log_message(f"Training for {EPOCHS} epochs")

    best_val_acc = -1
    best_epoch_name_str = None

    epoch_runner = EpochRunner(model, device, train_logger, optimizer, compound_scheduler, ce_loss, EPOCHS)
    for epoch in range(EPOCHS):
        train_accuracy = epoch_runner.run_epoch('train', epoch, train_dataloader)
        val_accuracy, _, _, _ = epoch_runner.run_epoch('val', epoch, val_dataloader)

        save_weights(model, f"epoch_{epoch+1:03d}.pth", WORK_DIR)
        if val_accuracy>best_val_acc:
            best_epoch_name_str = f"best_val_results_epoch_{epoch+1:03d}_acc_{val_accuracy:.2f}.pth"
            save_weights(model, 
                         best_epoch_name_str,
                         WORK_DIR,
                         "best_val_results")
            best_val_acc = val_accuracy

    train_logger.log_message(f"Training completed")
    train_logger.log_message(f"Best validation accuracy: {best_val_acc}")
    train_logger.log_message(f"To perform inference with the best model, use the following command:")
    train_logger.log_message(f"python test.py --model {MODEL} --weights {WORK_DIR}/{best_epoch_name_str} --data_path {DATA_PATH}  --output_folder {OUT_FOLDER} --exp_name {EXP_NAME}")