import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import matplotlib
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import matplotlib.lines as mlines
import argparse
import os.path as osp

def set_ax_borders(ax, top, right, bottom, left):
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", help="The model used for recognition", type=str)
    args = parser.parse_args()
    FILE = args.eval_file
    OUTPUT_FILE = osp.join(osp.dirname(FILE), "cm.png")


    with open(FILE, 'rb') as f:
        eval_file = pickle.load(f)

    gt_labels = np.array([i.item() for i in eval_file['gt_label']])
    pred_labels = np.array([i.item() for i in eval_file['pred_label']])

    POSTURE_LABEL2IDS = {'Supine': 0, 'Prone': 1, 'Sitting': 2, 'Standing': 3, 'All-fours': 4}
    POSTURE_IDS2LABEL = {v:k for k,v in POSTURE_LABEL2IDS.items()}

    def get_acc(gts, preds):
        assert len(preds)==len(gts), 'Predictions and Ground truths lenghts are incompatible'
        return (preds==gts).sum()/len(gts)

    def plot_cm(gts, preds, ax=None):
        acc = get_acc(gt_labels, pred_labels)
        labels = list(POSTURE_IDS2LABEL.values())
        cm = ConfusionMatrixDisplay.from_predictions(y_true=gts, y_pred=preds, normalize='true', display_labels=labels, ax=ax, values_format='.2f', cmap='Blues')
        cm.ax_.set_title(f"Accuracy: {acc*100:.2f}%")
        return cm

    COLORS = ['blue', 'orange', 'green', 'red', 'purple']
    CLASS_LABELS = ['Supine', 'Prone', 'Sitting', 'Standing',  'All-fours']

    fig, ax = plt.subplots()
    fig.set_size_inches(6,6)
    fig.set_dpi(600)
    plot_cm(gt_labels, pred_labels, ax=ax)
    plt.savefig(OUTPUT_FILE)
