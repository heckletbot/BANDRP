import os

from config import get_cfg_defaults
from data_load import *
from data_process import *
from model import *
from utils import *

# ----config
cfg = get_cfg_defaults()
if not os.path.exists(cfg['path']['savedir']):
    os.makedirs(cfg['path']['savedir'])
set_seed(2020)


def train(model, train_set, optimizer, myloss):
    model.train()
    predict_list = []
    label_list = []
    for batch, (drug_data, exp, mut, meth, path, label) in enumerate(train_set):
        drug_fp = [drug_data[i].to(device) for i in range(len(drug_data))]

        exp = exp.to(device)
        mut = mut.to(device)
        meth = meth.to(device)
        path = path.to(device)
        label = label.to(device)

        predict, _ = model(drug_fp, [exp, mut, meth, path])

        optimizer.zero_grad()
        loss = myloss(predict, label)
        loss.backward()
        optimizer.step()
        predict_list = predict_list + predict.flatten().tolist()
        label_list = label_list + label.flatten().tolist()
    train_loss = myloss(torch.tensor(predict_list), torch.tensor(label_list))
    return train_loss.item()


def test(model, test_set, myloss):
    model.eval()
    predict_list = []
    label_list = []
    with torch.no_grad():
        for batch, (drug_data, exp, mut, meth, path, label) in enumerate(test_set):
            drug_fp = [drug_data[i].to(device) for i in range(len(drug_data))]

            exp = exp.to(device)
            mut = mut.to(device)
            meth = meth.to(device)
            path = path.to(device)
            label = label.to(device)

            predict, _ = model(drug_fp, [exp, mut, meth, path])

            predict_list = predict_list + predict.flatten().tolist()
            label_list = label_list + label.flatten().tolist()
        test_loss = myloss(torch.tensor(predict_list), torch.tensor(label_list))
        mse, rmse, mae, r2, pearson, pearson_p_value, spearman, spearman_p_value = eval_predict(label_list,
                                                                                                predict_list)

    return test_loss.item(), mse, rmse, mae, r2, pearson, pearson_p_value, spearman, spearman_p_value


# data load
drug_feature, mut_feature, exp_feature, methy_feature, pathway_feature, pair, depmap_id, drug_id = dataload(**cfg)
print("load %s drugs and %s cell lines. total %s pairs" % (len(drug_id), len(depmap_id), len(pair)))
# set device
device = torch.device('cuda:%s' % cfg['model']['cuda_id'] if torch.cuda.is_available() else "cpu")

# split train and test sets
train_set, test_set, val_set = data_process(drug_feature, mut_feature, exp_feature, methy_feature, pathway_feature,
                                             pair,
                                             depmap_id, drug_id)
# create model
model = BANDRP(cell_exp_dim=exp_feature.shape[-1], cell_mut_dim=mut_feature.shape[-1],
            cell_meth_dim=methy_feature.shape[-1], cell_path_dim=pathway_feature.shape[-1], **cfg).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg['model']['lr'], weight_decay=cfg['model']['weight_decay'])
myloss = nn.MSELoss()
# train
min_mse = 100
train_loss = []
test_loss = []
best_eval = []
for epoch in range(cfg['model']['epoch']):
    epoch_train_loss = train(model, train_set, optimizer, myloss)
    epoch_test_loss, mse, rmse, mae, r2, pearson, pearson_p_value, spearman, spearman_p_value = test(model, val_set,
                                                                                                     myloss)
    critier = [epoch_train_loss, epoch_test_loss, mse, rmse, mae, r2, pearson, pearson_p_value, spearman,
               spearman_p_value]
    row = [str(i)[:8] for i in critier]
    row = [str(epoch)] + row
    table = print_table(row)
    train_loss.append(epoch_train_loss)
    test_loss.append(epoch_test_loss)
    if min_mse > mse:
        min_mse = mse
        torch.save(model.state_dict(), (cfg['path']['savedir'] + '/model.pt'))
        print("save!")
        best_eval = table
save_ouput(train_loss, test_loss, best_eval, cfg['path']['savedir'], 0)
draw_loss_curve(train_loss, test_loss, (cfg['path']['savedir'] + '/loss_curve.png'))
# test
model = BANDRP(cell_exp_dim=exp_feature.shape[-1], cell_mut_dim=mut_feature.shape[-1],
            cell_meth_dim=methy_feature.shape[-1], cell_path_dim=pathway_feature.shape[-1], **cfg).to(device)
model.load_state_dict(torch.load((cfg['path']['savedir'] + '/model.pt')))
epoch_test_loss, mse, rmse, mae, r2, pearson, pearson_p_value, spearman, spearman_p_value = test(model, test_set,
                                                                                                 myloss)
critier = [mse, rmse, mae, r2, pearson, pearson_p_value, spearman,
           spearman_p_value]
row = [str(i)[:8] for i in critier]
table = print_table(row, only_test=True)
