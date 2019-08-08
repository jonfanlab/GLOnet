"""General utility functions"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import json
import logging
import csv
import scipy.io as io
import torch
import numpy as np

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def row_csv2dict(csv_file):
    dict_club={}
    with open(csv_file)as f:
        reader=csv.reader(f,delimiter=',')
        for row in reader:
            dict_club[(row[0],row[1])]=row[2]
    return dict_club


def save_checkpoint(state, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'model.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)



def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)

    model.load_state_dict(checkpoint['gen_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def plot_loss_history(loss_history, params):
    effs_mean_history, diversity_history, binarization_history = loss_history
    iterations = [i*params.plot_iter for i in range(len(effs_mean_history))]
    print(iterations)
    plt.figure()
    plt.plot(iterations, effs_mean_history)
    plt.plot(iterations, diversity_history)
    plt.plot(iterations, binarization_history)
    plt.xlabel('iteration')
    plt.legend(('Average Efficiency', 'Pattern diversity', 'Binarizaion'))
    plt.axis([0, len(effs_mean_history), 0, 1.05])
    plt.savefig(params.output_dir + '/figures/Train_history.png')

    history_path = os.path.join(params.output_dir,'history.mat')
    io.savemat(history_path, mdict={'effs_mean_history':np.asarray(effs_mean_history), 
                                    'diversity_history':np.asarray(diversity_history),
                                    'binarization_history'  :np.asarray(binarization_history)})
         
def plot_scatter(imgs, Effs, Iter, fig_path):
    fig = plt.figure()
    plt.scatter(imgs[:, 0], imgs[:, 1], c = Effs*100, cmap=plt.cm.rainbow, vmin=0, vmax=100)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.clim(0, 100)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.yticks([])
    plt.xticks([])    
    #plt.xticks(fontsize=20)
    #plt.yticks(fontsize=20)
    plt.title('Iteration {}'.format(Iter), fontsize=16)
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plot_scatter_and_histogram(imgs, Effs, Iter, fig_path):
    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1]) 
    plt.suptitle('Iteration {}'.format(Iter), fontsize=16)

    plt.subplot(gs[0])
    plt.scatter(imgs[:, 0], imgs[:, 1], c = Effs*100, cmap=plt.cm.rainbow, vmin=0, vmax=100)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.clim(0, 100)
    plt.xlim(-10, 5)
    plt.ylim(0, 10)
    plt.yticks([])
    plt.xticks([])    
    #plt.xticks(fontsize=20)
    #plt.yticks(fontsize=20)

    plt.subplot(gs[1])
    bins = [i*5 for i in range(21)]
    plt.hist(Effs*100, bins, facecolor='blue', alpha=0.5)
    plt.xlim(0, 100)
    plt.ylim(0, 50)
    plt.yticks([])
    #plt.xticks(fontsize=20)
    plt.xticks(fontsize=12)
    plt.xlabel('Deflection efficiency (%)', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(fig_path, dpi=300)
    plt.close()

def plot_histogram(Effs, Iter, fig_path):
    ax = plt.figure()
    bins = [i*5 for i in range(21)]
    plt.hist(Effs*100, bins, facecolor='blue', alpha=0.5)
    plt.xlim(0, 100)
    plt.ylim(0, 50)
    plt.yticks([])
    plt.xticks(fontsize=12)
    #plt.yticks(fontsize=20)
    plt.xlabel('Deflection efficiency (%)', fontsize=12)
    plt.title('Iteration {}'.format(Iter), fontsize=16)
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plot_arrow(imgs, Effs, grads, Iter, fig_path):
    ax = plt.figure()
    plt.scatter(imgs[:, 0], imgs[:, 1], c = Effs*100, cmap=plt.cm.rainbow, vmin=0, vmax=100)
    plt.colorbar()
    plt.clim(0, 100)
    plt.quiver(imgs[:, 0], imgs[:, 1], grads[:, 0], grads[:, 1])
    #plt.xlim(-11, -3)
    #plt.ylim(2, 8)
    #plt.xticks(fontsize=20)
    #plt.yticks(fontsize=20)
    #plt.title('Iteration {}'.format(Iter), fontsize=20)
    plt.yticks([])
    plt.xticks([])    
    plt.savefig(fig_path, dpi=300)
    plt.close()

def plot_envolution(imgs_prev, Effs_prev, grads_prev, imgs, Effs, Iter, fig_path):
    ax = plt.figure(figsize=(3, 3))

    Effs_prev = np.ones_like(Effs_prev)*0.2
    Effs = np.ones_like(Effs)
    plt.scatter(imgs_prev[:, 0], imgs_prev[:, 1], c = Effs_prev, cmap=plt.cm.rainbow, vmin=0, vmax=1)
    plt.scatter(imgs[:, 0], imgs[:, 1], c = Effs, cmap=plt.cm.rainbow, vmin=0, vmax=1)
    plt.quiver(imgs_prev[:, 0], imgs_prev[:, 1], grads_prev[:, 0], grads_prev[:, 1])
    #plt.colorbar()
    #plt.clim(0, 1)
    #plt.xlim(-11, -3)
    #plt.ylim(2, 8)
    #plt.xticks(fontsize=20)
    #plt.yticks(fontsize=20)
    #plt.title('Iteration {}'.format(Iter), fontsize=20)
    plt.yticks([])
    plt.xticks([])    
    plt.savefig(fig_path, dpi=300)
    plt.close()



def movie_scatter(imgs, Effs, output_dir):

    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='scatter', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=5, metadata=metadata)

    fig = plt.figure()
    numFrame = imgs.shape[0]
    filepath = output_dir + '/scatter.avi'

    with writer.saving(fig, filepath, numFrame):
        for i in range(numFrame):
            plt.cla()
            plt.scatter(imgs[i, :, 0], imgs[i, :, 1], c = Effs[i, :], cmap=plt.cm.plasma)
            plt.title('Iter {}'.format(i*100+100))
            plt.xlim(-15, 15)
            plt.ylim(-15, 15)
            plt.colorbar()
            plt.clim(0, 1)
            writer.grab_frame()


