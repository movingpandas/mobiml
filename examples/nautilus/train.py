import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# import pdb
import tqdm
import time
import torch
import torch.nn as nn

ROUND_DECIMALS = 5


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps
        print(f'{self.eps=}')

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def save_model(model, path, **kwargs):
    torch.save({
        'model_state_dict': model.state_dict(),
        **kwargs
    }, path)


def calc_loss(model, xb, yb, criterion, *args):
    y_pred = model(xb, *args)
    loss = criterion(y_pred, yb.squeeze(dim=-2))
    return y_pred, loss
    

def model_backprop(model, xb, yb, criterion, optimizer, *args):
    try:
        optimizer.zero_grad()
        _, loss = calc_loss(model, xb, yb, criterion, *args)

        loss.backward()
        optimizer.step()
    except RuntimeError as err_runtime:
        print(err_runtime)
        # pdb.set_trace()
    return loss


def running_loss(loss, data_loader):
    loss = torch.Tensor(loss).sum()
    loss = loss / len(data_loader)
    return loss


def model_dev_loss(model, device, criterion, dev_loader):
    model.eval()
    with torch.no_grad():
        dev_loss = []
        for (xb, yb, lb, *args) in (pbar := tqdm.tqdm(dev_loader, leave=False, total=len(dev_loader), dynamic_ncols=True)):
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            args = (arg.to(device) for arg in args)
            
            _, loss = calc_loss(model, xb, yb, criterion, lb, *args)
            dev_loss.append(loss)

            pbar.set_description(f'Dev Loss: {loss:.{ROUND_DECIMALS}f}')

    return running_loss(dev_loss, dev_loader)


def train_step(model, device, criterion, optimizer, train_loader):
    model.train()

    train_loss = []
    for j, (xb, yb, lb, *args) in (pbar := tqdm.tqdm(enumerate(train_loader), leave=False, total=len(train_loader), dynamic_ncols=True)):
        xb = xb.to(device).float()
        yb = yb.to(device).float()
        args = (arg.to(device) for arg in args)

        tr_loss = model_backprop(model, xb, yb, criterion, optimizer, lb, *args)
        train_loss.append(tr_loss)
        pbar.set_description(f'Train Loss: {tr_loss:.{ROUND_DECIMALS}f}')

    return running_loss(train_loss, train_loader)


def early_stopping(n_epochs_stop, min_loss, curr_loss, patience=5, min_delta=1e-4, save_best=False, **kwargs):
    if (min_loss - curr_loss) > min_delta:
        if save_best:
            print(f'Loss Decreased ({min_loss:.{ROUND_DECIMALS}f} -> {curr_loss:.{ROUND_DECIMALS}f}). Saving Model...', end=' ')
            save_model(**kwargs)
            print('Done!')

        return 0, curr_loss, False

    print(f'Loss Increased ({min_loss:.{ROUND_DECIMALS}f} -> {curr_loss:.{ROUND_DECIMALS}f}).')
    n_epochs_stop_ = n_epochs_stop + 1
    return n_epochs_stop_, min_loss, n_epochs_stop_ == patience


def vrf_evaluate_model_singlehead(model, device, criterion, test_loader, display_acc=True,
                                  bins=np.arange(0, 1801, 300), desc=None, **kwargs):
    errs, losses = [], []

    model.eval()
    with torch.no_grad():
        for xb, yb, lb, *args in (pbar := tqdm.tqdm(test_loader, leave=False, desc=desc, total=len(test_loader), dynamic_ncols=True)):
            # print(f'{xb.shape=}\t {yb.shape=}\t {lb.shape=}')
            xb, yb = xb.to(device), yb.squeeze(dim=-2).to(device)    # Model Inference
            args = (arg.to(device) for arg in args)
            y_pred = model(xb.float(), lb, *args).detach()

            errs.append(pd.DataFrame({'errs':np.linalg.norm(y_pred.cpu() - yb.cpu(), axis=-1)}))
            losses.append(eval_loss := criterion(y_pred, yb))
            pbar.set_description(f'{desc}: {eval_loss:.{ROUND_DECIMALS}f}')

    # pdb.set_trace()
    errs = pd.concat(errs, ignore_index=True)
    errs.loc[:, 'look'] = [i[-1, -1] for i in test_loader.dataset.samples]

    test_loss = running_loss(losses, test_loader)
    avg_disp_err = np.mean(errs.errs)

    if display_acc:
        look_bins_cut = pd.cut(errs.look, bins, include_lowest=True)
        ade_bins_cut = errs.groupby(look_bins_cut).errs.mean()
        # ade_bins_cut = errs.groupby(look_bins_cut).agg({'errs': lambda x: x.mean(skipna=False)}).errs

        # Avg. Loss | Avg. Disp. Error (@5 min.; @10 min.; @15 min.; @20 min.; @25 min.; @30 min.; @35 min.)
        print(f'Loss: {test_loss:.{ROUND_DECIMALS}f} | ', f'Accuracy: {avg_disp_err:.{ROUND_DECIMALS}f} |', '; '.join(f'{i:.{ROUND_DECIMALS}f}' for i in ade_bins_cut.values.tolist()), 'm')

    return test_loss, avg_disp_err


def train_model(model, device, criterion, optimizer, n_epochs,
                train_loader, dev_loader, evaluate_cycle=5, early_stop=True, save_current=True,
                evaluate_fun=vrf_evaluate_model_singlehead, evaluate_fun_params={}, early_stop_params={}, save_current_params={}):
    train_losses, dev_losses = [], []

    # Early Stopping Initial Param. Values
    min_loss, n_epochs_stop, stop = early_stop_params.pop('min_loss', torch.tensor(float("Inf"))), 0, False

    if save_current:
        save_path_template = save_current_params['path']

    # training loop
    for i in range(n_epochs):
        t_start = time.process_time()
        train_loss = train_step(model, device, criterion, optimizer, train_loader)
        dev_loss = model_dev_loss(model, device, criterion, dev_loader)
        t_end = time.process_time() - t_start

        train_losses.append(train_loss.numpy())
        dev_losses.append(dev_loss.numpy())

        epoch_summary = {
            'model': model,
            'epoch': i,
            'scaler': train_loader.dataset.scaler,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_losses,
            'dev_loss': dev_losses
        }

        if early_stop:
            early_stop_params.update(epoch_summary)
            n_epochs_stop, min_loss, stop = early_stopping(n_epochs_stop, min_loss, dev_loss, **early_stop_params)
        
        if save_current:
            save_current_params.update(epoch_summary)
            save_current_params['path'] = save_path_template.format(i)
            save_model(**save_current_params)

        print(f'Epoch #{i+1}/{n_epochs} | '
              f'Train Loss: {train_loss:.{ROUND_DECIMALS}f} | '
              f'Validation Loss: {dev_loss:.{ROUND_DECIMALS}f} | '
              f'Time Elapsed: {t_end:.{ROUND_DECIMALS}f}')

        if evaluate_cycle != -1 and i % evaluate_cycle == 0:
            evaluate_fun(model, device, criterion, dev_loader,
                         desc='ADE @ Dev Set...', **evaluate_fun_params)

        if stop:
            print(f'Training Stopped at Epoch #{i+1}')
            break
    return train_losses, dev_losses
