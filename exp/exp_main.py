from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
import models.DMamba as DMamba
import models.DMamba_Hybrid as DMamba_Hybrid
import models.DMamba_T as DMamba_T
import models.DMamba_AllMamba as DMamba_AllMamba
import models.DMamba_MixedMamba as DMamba_MixedMamba
import models.DMamba_TrendMamba as DMamba_TrendMamba
import models.DMamba_TMamba as DMamba_TMamba
import models.DMamba_DualMamba as DMamba_DualMamba
import models.DMamba_MLP as DMamba_MLP
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import math

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'DMamba': DMamba,
            'DMamba_hybrid': DMamba_Hybrid,
            'DMamba_T': DMamba_T,
            'DMamba_AllMamba': DMamba_AllMamba,
            'DMamba_MixedMamba': DMamba_MixedMamba,
            'DMamba_TrendMamba': DMamba_TrendMamba,
            'DMamba_TMamba': DMamba_TMamba,
            'DMamba_DualMamba': DMamba_DualMamba,
            'DMamba_MLP': DMamba_MLP,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # # MSE criterion
    # def _select_criterion(self):
    #     criterion = nn.MSELoss()
    #     return criterion

    # MSE and MAE criterion
    def _select_criterion(self):
        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        return mse_criterion, mae_criterion

    def vali(self, vali_data, vali_loader, mse_criterion, mae_criterion, is_test=True):
        total_mse = []
        total_mae = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # if train, use ratio to scale the prediction
                if not is_test:
                    # Arctangent loss with weight decay
                    self.ratio = np.array([-1 * math.atan(i+1) + math.pi/4 + 1 for i in range(self.args.pred_len)])
                    self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')
                    pred = outputs*self.ratio
                    true = batch_y*self.ratio
                else:
                    pred = outputs
                    true = batch_y

                mse = mse_criterion(pred, true)
                mae = mae_criterion(pred, true)

                total_mse.append(mse.item())
                total_mae.append(mae.item())

        total_mse = np.average(total_mse)
        total_mae = np.average(total_mae)
        self.model.train()
        return total_mse, total_mae

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        # criterion = self._select_criterion() # For MSE criterion
        mse_criterion, mae_criterion = self._select_criterion()

        # # CARD's cosine learning rate decay with warmup
        # self.warmup_epochs = self.args.warmup_epochs

        # def adjust_learning_rate_new(optimizer, epoch, args):
        #     """Decay the learning rate with half-cycle cosine after warmup"""
        #     min_lr = 0
        #     if epoch < self.warmup_epochs:
        #         lr = self.args.learning_rate * epoch / self.warmup_epochs 
        #     else:
        #         lr = min_lr+ (self.args.learning_rate - min_lr) * 0.5 * \
        #             (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.args.train_epochs - self.warmup_epochs)))
                
        #     for param_group in optimizer.param_groups:
        #         if "lr_scale" in param_group:
        #             param_group["lr"] = lr * param_group["lr_scale"]
        #         else:
        #             param_group["lr"] = lr
        #     print(f'Updating learning rate to {lr:.7f}')
        #     return lr

        # train_times = [] # For computational cost analysis
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_mse = []
            train_mae = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Arctangent loss with weight decay
                self.ratio = np.array([-1 * math.atan(i+1) + math.pi/4 + 1 for i in range(self.args.pred_len)])
                self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')

                outputs = outputs * self.ratio
                batch_y = batch_y * self.ratio

                loss_mse = mse_criterion(outputs, batch_y)
                loss_mae = mae_criterion(outputs, batch_y)

                # Backward with MAE
                loss_mae.backward()
                
                model_optim.step()

                train_mse.append(loss_mse.item())
                train_mae.append(loss_mae.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_mse.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_mse = np.average(train_mse)
            train_mae = np.average(train_mae)

            vali_mse, vali_mae = self.vali(vali_data, vali_loader, mse_criterion, mae_criterion, is_test=False)
            test_mse, test_mae = self.vali(test_data, test_loader, mse_criterion, mae_criterion, is_test=True)

            print("Epoch: {0}, Steps: {1} | Train MSE: {2:.7f} MAE: {3:.7f} Vali MSE: {4:.7f} MAE: {5:.7f} Test MSE: {6:.7f} MAE: {7:.7f}".format(
                epoch + 1, train_steps, train_mse, train_mae, vali_mse, vali_mae, test_mse, test_mae))
            early_stopping(vali_mse, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            # adjust_learning_rate_new(model_optim, epoch + 1, self.args)

            # print('Alpha:', self.model.decomp.ma.alpha) # Print the learned alpha
            # print('Beta:', self.model.decomp.ma.beta)   # Print the learned beta

        # print("Training time: {}".format(np.sum(train_times)/len(train_times))) # For computational cost analysis
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        os.remove(best_model_path)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # test_time = 0 # For computational cost analysis
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                # temp = time.time() # For computational cost analysis
                outputs = self.model(batch_x)
                # test_time += time.time() - temp # For computational cost analysis

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
            
        # print("Inference time: {}".format(test_time/len(test_loader))) # For computational cost analysis
        preds = np.array(preds)
        trues = np.array(trues)
        # preds = np.concatenate(preds, axis=0) # without the "drop-last" trick
        # trues = np.concatenate(trues, axis=0) # without the "drop-last" trick

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return
