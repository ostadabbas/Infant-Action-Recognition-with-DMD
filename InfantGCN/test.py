import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
import tqdm
import os
import os.path as osp
from datetime import datetime
import pickle

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
    parser.add_argument("--weights", help="The model used for recognition", type=str)
    parser.add_argument("--data_path", help="The data used for recognition", type=str)
    parser.add_argument("--output_folder", help="output folder to save the results", type=str)
    parser.add_argument("--exp_name", help="name of the experiments", type=str)
    return parser

def do_epoch(model, dataloader):
    torch_mode = torch.no_grad
    model.eval()

    correct_predictions = 0
    total_predictions = 0
    all_preds  = []
    all_gts = []
    all_feats = []
    with torch_mode():
        for batch_X, batch_y in tqdm.tqdm(dataloader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_pred, batch_feats = model.extract_features(batch_X)

            _, predicted_labels = torch.max(batch_pred, 1)
            correct_predictions += (predicted_labels == batch_y).sum().item()
            total_predictions += batch_y.size(0)
            all_preds.extend(predicted_labels.cpu().numpy())
            all_gts.extend(batch_y.cpu().numpy())
            all_feats.extend(batch_feats.cpu().numpy())
        accuracy = correct_predictions / total_predictions

        print(f"test acc: {accuracy:.4f}")
    return accuracy, all_preds, all_gts, all_feats

if __name__ == "__main__":
    parser = get_parsers()
    args = parser.parse_args()

    MODEL = args.model
    WEIGHTS = args.weights
    EXP_NAME = args.exp_name
    DATA_PATH = args.data_path
    OUT_FOLDER = args.output_folder if args.output_folder is not None else "../Results_test"
    WORK_DIR = osp.join(OUT_FOLDER, EXP_NAME)

    N_FEATS = 2

    # test_dataset = Feeder(f"../Data/InfAct_plus/InfAct_plus_{N_FEATS}d_yolo.pkl", 'test', window_size=60, random_selection="uniform_choose")
    test_dataset = Feeder(DATA_PATH, 'test', window_size=60, random_selection="uniform_choose", break_samples=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    graph_args = {'layout': f'infant{N_FEATS}d', 'strategy': 'spatial'}
    kwargs = {}
    input_dim = N_FEATS
    num_classes = 5
    if MODEL=="STGCN":
        model = STGCN(input_dim, num_classes, graph_args, edge_importance_weighting = True, **kwargs).to(device)
    elif MODEL=="CTRGCN":
        model = CTRGCN(input_dim, num_classes, graph_args, **kwargs).to(device)

    model.load_state_dict(torch.load(WEIGHTS))

    test_accuracy, preds, gts, feats = do_epoch(model, test_dataloader)
    with open(osp.join(WORK_DIR, "eval.pkl"), "wb") as f:
        pickle.dump({"Accuracy": test_accuracy, "pred_label": preds, "gt_label": gts, "feats": feats}, f)