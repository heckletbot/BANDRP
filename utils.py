import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from prettytable import PrettyTable
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


def eval_predict(y_label, y_pred):
    # MAE MSE RMSE R^2
    mae = mean_absolute_error(y_label, y_pred)
    mse = mean_squared_error(y_label, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_label, y_pred)

    # pearson spearman
    pearson = pearsonr(y_label, y_pred)[0]
    pearson_p_value = pearsonr(y_label, y_pred)[1]
    spearman = spearmanr(y_label, y_pred)[0]
    spearman_p_value = spearmanr(y_label, y_pred)[1]

    return mse, rmse, mae, r2, pearson, pearson_p_value, spearman, spearman_p_value


def save_ouput(train_loss, test_loss, best_model_eval, savedir, settype):
    if settype == "val":
        savedir_ = (savedir + '/fold%s') % (settype)
        np.save(savedir_ + 'train_loss.npy', train_loss)
        np.save(savedir_ + 'test_loss.npy', test_loss)
    record_file = savedir+'/best_model.txt'
    with open(record_file, 'a') as f:
        f.write(("\ndataset:%s \n" % (settype)))
        f.write(str(best_model_eval))
        f.close()
    return


def draw_loss_curve(train_loss, test_loss, savedir):
    plt.clf()
    plt.plot(np.arange(len(train_loss)), train_loss, label="train loss")
    plt.plot(np.arange(len(test_loss)), test_loss, label="test loss")
    plt.legend()  
    plt.xlabel('epoches')
    plt.title('Model loss')
    plt.show()
    plt.savefig(savedir + 'loss.png',
                dpi=100,
                facecolor='violet',
                edgecolor='lightgreen',
                bbox_inches='tight')
    return


def print_table(data, only_test=False, title=None, headers=None):
    table = PrettyTable()
    if title:
        table.title = title
    if not only_test:
        table.field_names = ['epoch', 'train_loss', 'test_loss', 'mse', 'rmse', 'mae', 'r2', 'pearson', 'pcc-p',
                             'spearman', 'scc-p']
    else:
        print('test_set')
        table.field_names = ['mse', 'rmse', 'mae', 'r2', 'pearson', 'pcc-p',
                             'spearman', 'scc-p']
    table.add_row(data)
    print(table)
    return table
