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

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ChainedScheduler

from net.stgcn import STGCN
from net.ctrgcn import CTRGCN
from dataloader import Feeder

import argparse

def get_parsers():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CTRGCN", help="The model used for recognition", type=str)
    parser.add_argument("--base_lr", default=0.1, help="Base learing rate", type=float)
    parser.add_argument("--epochs", default=20, help="number of epochs to train the dataset", type=int)
    parser.add_argument("--repeat", default=1, help="number of times to repeat training dataset", type=int)
    parser.add_argument("--output_folder", help="output folder to save the results", type=str)
    parser.add_argument("--exp_name", help="name of the experiments", type=str)
    return parser

def save_weights(model, name):
    torch.save(model.state_dict(), osp.join(WORK_DIR, name))

def do_epoch(mode, epoch_num, model, dataloader):
    if mode == 'train':
        torch_mode = torch.enable_grad
        model.train()
        running_loss = 0
    else:
        torch_mode = torch.no_grad
        model.eval()

    correct_predictions = 0
    total_predictions = 0
    with torch_mode():
        for batch_X, batch_y in tqdm.tqdm(dataloader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_pred = model(batch_X)

            if mode=='train':
                loss = ce_loss(batch_pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()

            _, predicted_labels = torch.max(batch_pred, 1)
            correct_predictions += (predicted_labels == batch_y).sum().item()
            total_predictions += batch_y.size(0)
        accuracy = correct_predictions / total_predictions

        if mode=='train':
            epoch_loss = running_loss/len(dataloader)
            print(f"epoch: {epoch_num+1}, train_loss; {epoch_loss:.4f}, train acc: {accuracy:.4f}")
        else:
            print(f"epoch: {epoch_num+1}, val acc: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    parser = get_parsers()
    args = parser.parse_args()

    EPOCHS = args.epochs
    MODEL = args.model
    BASE_LR = args.base_lr
    REPEAT = args.repeat
    now = datetime.now().strftime("%m-%d-%y_%H-%M-%S")
    EXP_NAME = args.exp_name if args.exp_name is not None else f"exp_{now}"
    OUT_FOLDER = args.output_folder if args.output_folder is not None else "../Results_test"
    WORK_DIR = osp.join(OUT_FOLDER, EXP_NAME)
    os.makedirs(WORK_DIR, exist_ok=True)

    
    N_FEATS = 2

    train_dataset = Feeder(f"../Data\\InfAct_plus\\InfAct_plus_{N_FEATS}d_yt_split.pkl", 'train', window_size=60, random_selection="uniform_choose", repeat=REPEAT)
    val_dataset = Feeder(f"../Data\\InfAct_plus\\InfAct_plus_{N_FEATS}d_yt_split.pkl", 'val', window_size=60, random_selection="uniform_choose")

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=len(train_dataloader)*40)
    scheduler2 = CosineAnnealingLR(optimizer, eta_min=0, T_max = len(train_dataloader)*60)
    scheduler = ChainedScheduler([scheduler1, scheduler2])

    best_acc = -1
    for epoch in range(EPOCHS):
        train_accuracy = do_epoch('train', epoch, model, train_dataloader)
        test_accuracy = do_epoch('test', epoch, model, val_dataloader)
        save_weights(model, f"epoch_{epoch+1}.pth")
        if test_accuracy>best_acc:
            save_weights(model, "best_results.pth")
            best_acc = test_accuracy