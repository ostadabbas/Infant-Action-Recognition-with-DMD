import numpy as np
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# visualization
import time

# operation
from . import tools

POSTURE_CODEBOOK = {0: 'Supine', 1: 'Prone', 2: 'Sitting', 3: 'Standing', 4: 'All-fours',5: 'Transition'}
POSLABEL2LABEL = {v:k for k,v in POSTURE_CODEBOOK.items()}

class Feeder(torch.utils.data.Dataset):
    
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 fold,
                 random_selection=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 repeat=1):
        self.debug = debug
        self.data_path = data_path
        self.fold = fold
        self.random_selection = random_selection
        self.random_move = random_move
        self.window_size = window_size
        self.repeat = repeat

        self.load_data()

    def load_data(self):
        # data: N C V T M
        with open(self.data_path, 'rb') as f:
            file = pickle.load(f)
        fold_files = [item for item in file['annotations'] if item['frame_dir'] in file['split'][self.fold]]
        self.data = [item['keypoint'].transpose(3,1,2,0) for item in fold_files]
        self.sample_name = [item['frame_dir'] for item in fold_files]
        try:
            self.label  = [item['label'] for item in fold_files]
        except:
            self.label  = [POSLABEL2LABEL[item['pos_label']] for item in fold_files]
            
        if self.debug:
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        #self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return self.repeat*len(self.label)

    def __getitem__(self, index):
        # get data
        index = index%len(self.label)
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        
        # processing
        if self.random_selection is not None:
            if self.random_selection == "random_choose":
                data_numpy = tools.random_choose(data_numpy, self.window_size)
            elif self.random_selection == "uniform_choose":
                data_numpy = tools.uniform_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        data_numpy - data_numpy[:,0,0,0]

        return data_numpy.astype(np.float32), label