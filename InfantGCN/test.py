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
from utils.logger import Logger
from utils.parsers import get_testing_parser
from epoch_runner import EpochRunner

if __name__ == "__main__":
    parser = get_testing_parser()
    args = parser.parse_args()

    MODEL = args.model
    WEIGHTS = args.weights
    EXP_NAME = args.exp_name
    DATA_PATH = args.data_path
    OUT_FOLDER = args.output_folder if args.output_folder is not None else "../Results_test"
    WORK_DIR = osp.join(OUT_FOLDER, EXP_NAME)

    N_FEATS = 2

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
    test_logger = Logger('train', osp.join(WORK_DIR, 'test_log.txt'))
    epoch_runner = EpochRunner(model, device)
    test_accuracy, test_loss, preds, gts, feats = epoch_runner.run_epoch('val', 0, test_dataloader)
    test_logger.log_test(1,test_loss,test_accuracy)
    with open(osp.join(WORK_DIR, "eval.pkl"), "wb") as f:
        pickle.dump({"Accuracy": test_accuracy, "pred_label": preds, "gt_label": gts, "feats": feats}, f)

    test_logger.log_message(f"Testing complete")
    test_logger.log_message(f"Test Accuracy: {test_accuracy}")