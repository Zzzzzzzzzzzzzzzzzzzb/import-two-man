from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Tide_Pred
from torch.utils.data import DataLoader
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, DLL, Tide
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from visdom import Visdom

warnings.filterwarnings('ignore')


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.sqrt(torch.mean(torch.pow(torch.log(x + 1) - torch.log(y + 1), 2)))


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.index = 0
        self.split_len_1 = 0
        self.split_len_2 = 0

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'DLL': DLL,
            'Tide': Tide
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, index=None):
        data_set, data_loader = data_provider(self.args, flag, index=index)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
        # criterion = My_loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
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
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'DLL' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'Tide' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def get_index_list(self):
        path = self.args.root_path + self.args.data_path
        df = pd.read_csv(path)
        res = []
        if self.args.if_id_or_not:
            id = df.drop_duplicates(subset=['id'])
            id = list(id['id'])
            for i in id:
                temp = df[df['id'] == i]
                if len(temp) >= self.args.seq_len + self.args.pred_len:
                    index = list(temp.index)
                    res += index[:-self.args.pred_len-self.args.seq_len+1]
        else:
            print(f'样本长度为{len(df)}，开始采样 =================================')
            windows_len = self.args.seq_len + self.args.pred_len
            i = 0
            while i < len(df)-windows_len:
                if df.loc[i+windows_len, 'date'] < df.loc[i, 'date']:
                    i += windows_len
                    continue
                res.append(i)
                i += 1
                if i % 10000 == 0:
                    print(f'已采样第{i}个')
        return res

    def train(self, setting):
        self.index = self.get_index_list()
        a = self.index[-1]
        self.split_len_1 = int(len(self.index)*0.8)
        self.split_len_2 = int(len(self.index)*0.9)

        train_index = self.index[: self.split_len_1]
        train_data, train_loader = self._get_data(flag='train', index=train_index)
        vali_index = self.index[self.split_len_1: self.split_len_2]
        vali_data, vali_loader = self._get_data(flag='val', index=vali_index)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        viz = Visdom()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            loop = tqdm(train_loader, desc='Train', colour='blue')
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loop):
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
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'DLL' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'Tide' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                loop.set_description(f'Epoch [{epoch+1}/{self.args.train_epochs}]')
                loop.set_postfix(loss=loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()

                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 实时loss监控
            if epoch == 0:
                viz.line([[train_loss, vali_loss]], [0], win='Loss_each_epoch',
                         opts=dict(title='Loss each epoch', legend=['train loss', 'vali loss']))
            else:
                viz.line([[train_loss, vali_loss]], [epoch], win='Loss_each_epoch', update='append')

            print(f"Epoch: {epoch + 1} || cost time: {time.time() - epoch_time} || train loss: {train_loss} "
                  f"|| Vali Loss: {vali_loss}")

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

    def test(self, setting, test=1):
        test_index = self.index[self.split_len_2:]
        test_data, test_loader = self._get_data(flag='test', index=test_index)
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(f'./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []

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
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'DLL' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'Tide' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save ======================================================================
        # df_pred = pd.DataFrame()
        # for i in range(len(test_index)):
        #     print(f'============ {i} ============')
        #     date_range = list(test_data.time_stamp.iloc[test_index[i]+self.args.seq_len: test_index[i]+self.args.seq_len+self.args.pred_len, 0])
        #     date_range = np.transpose([date_range])
        #     id = np.transpose([[test_data.attributes[test_index[i]]] * self.args.pred_len])
        #     temp = pd.DataFrame(
        #         np.append(np.append(date_range, id, axis=1), preds[i], axis=1),
        #         columns=test_data.cols)
        #     df_pred = pd.concat([df_pred, temp]).reset_index(drop=True)
        #
        # folder_path = f'./results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        # df_pred.to_csv(folder_path + 'real_prediction.csv', index=False)

        mae, mse, rmse, mape, mspe, rse, corr, wmape = metric(preds, trues)
        print('mse:{}, mae:{}, wmape:{}'.format(mse, mae, wmape))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        return

    def predict(self, setting, load=False):
        self.index = self.get_index_list()
        if 'Tide' in self.args.model:
            pred_data = Dataset_Tide_Pred(root_path=self.args.root_path, data_path=self.args.data_path, index=self.index,
                                          size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
                                          freq=self.args.freq, enc_in=self.args.enc_in)
            pred_loader = DataLoader(pred_data, batch_size=self.args.batch_size, shuffle=False,
                                                   num_workers=self.args.num_workers, drop_last=False)
        else:
            pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                print(f'========================= {i} =========================')
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'DLL' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'Tide' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)

        df_pred = pd.DataFrame()
        for i in range(len(self.index)):
            print(f'============ {i} ============')
            date_range = list(pred_data.time_stamp.iloc[
                              self.index[i] + self.args.seq_len: self.index[i] + self.args.seq_len + self.args.pred_len,
                              0])
            date_range = np.transpose([date_range])
            id = np.transpose([[pred_data.attributes[self.index[i]]] * self.args.pred_len])
            temp = pd.DataFrame(
                np.append(np.append(date_range, id, axis=1), preds[i], axis=1),
                columns=pred_data.cols)
            df_pred = pd.concat([df_pred, temp]).reset_index(drop=True)

        folder_path = f'./results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # result save
        folder_path = './results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        df_pred.to_csv(folder_path + f'{self.args.model_id}_predict.csv', index=False)

        return



