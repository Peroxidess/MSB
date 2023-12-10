import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as Data
from preprocess.get_dataset import MyDataset


class Baseline():
    def __init__(self, model_name, train_set, train_label, val_set, val_label, test_set, test_label, target,
                 co_col, ca_col, task_name, seed, param_init={}, param_fit={}):
        self.train_set = train_set
        self.train_label = train_label
        self.val_set = val_set
        self.val_label = val_label
        self.test_set = test_set
        self.test_label = test_label
        self.target = target
        self.co_col = co_col
        self.ca_col = ca_col
        self.task_name = task_name
        self.seed = seed
        if 'class' in task_name:
            self.output_dim = len(np.unique(self.train_label.values))
        else:
            self.output_dim = 1
        self.model = self.model_bulid(model_name, param_init)
        self.param_grid = {}
        self.param_fit = param_fit

    @staticmethod
    def input_process(input_x, input_y):
        return input_x, input_y

    def model_bulid(self, model_name, param_init):
        model = eval(model_name)(**param_init)
        return model

    def grid_fit_pred(self):
        train_x, train_y = self.input_process(self.train_set, self.train_label)
        val_x, val_y = self.input_process(self.val_set, self.val_label)
        test_x, test_y = self.input_process(self.test_set, self.test_label)
        clf = GridSearchCV(self.model, self.param_grid)
        clf.fit(train_x, train_y[self.target['label1']], **self.param_fit)
        print('Best parameters found by grid search are:', clf.best_params_)
        self.model = clf.best_estimator_
        self.model.fit(train_x, train_y[self.target['label1']], **self.param_fit)
        pred_tra = self.model.predict(train_x).reshape(-1, 1)
        pred_val = self.model.predict(val_x).reshape(-1, 1)
        pred_test = self.model.predict(test_x).reshape(-1, 1)
        pred_tra_df = pd.DataFrame(pred_tra, index=self.train_label.index, columns=[self.target['label1']])
        pred_val_df = pd.DataFrame(pred_val, index=self.val_label.index, columns=[self.target['label1']])
        pred_test_df = pd.DataFrame(pred_test, index=self.test_label.index, columns=[self.target['label1']])
        return pred_tra_df, pred_val_df, pred_test_df, self.model

    def imp_feat(self):
        try:
            feat_dict = dict(zip(self.train_set.columns, self.model.coef_))
        except:
            feat_dict = dict(zip(self.train_set.columns, self.model.feature_importances_))
        else:
            pass
        return pd.DataFrame.from_dict([feat_dict], orient='columns').T.sort_values(by=0, ascending=False)


class MLP(Baseline):
    def __init__(self, model, train_set, train_label, val_set, val_label, test_set, test_label, target, co_col, ca_col, task_name, seed, param_init={}, param_fit={}):
        Baseline.__init__(self, model, train_set, train_label, val_set, val_label, test_set, test_label, target, co_col, ca_col, task_name, seed,
                          param_init, param_fit)
        self.param_fit = param_fit
        self.seed =seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def model_bulid(self, model_name, param_init):
        model = self.MLP_base(seed=self.seed, dim_input=self.train_set.shape[1], **param_init)
        return model

    def input_process(self, input_x, input_y, state=None):
        data_labeled = MyDataset(input_x, input_y)
        return data_labeled

    def grid_fit_pred(self, batch_size=64):
        train_data_set = self.input_process(self.train_set, self.train_label.iloc[:, 0])
        optim_ = optim.Adam(self.model.parameters(), lr=2e-4, weight_decay=1e-4)
        train_data_loader = Data.DataLoader(train_data_set, batch_size=batch_size, worker_init_fn=np.random.seed(self.seed))
        total_loss_list = []
        best_val_loss, es = 10., 0
        for iter_count in range(self.param_fit['epoch']):
            total_loss_ = 0.
            for train_data_batch, train_y_batch in train_data_loader: #labeled data loader is the same as unlabeled data loader here
                self.model.train()
                optim_.zero_grad()
                y, features = self.model(train_data_batch)
                label_onehot = torch.ones(train_y_batch.shape[0],
                            2
                            ).scatter_(1, train_y_batch.view(-1, 1), 0)
                CELoss = self.model.loss(y, label_onehot)
                total_loss = CELoss
                total_loss.backward()
                optim_.step()
                total_loss_ += total_loss.detach().numpy()

            self.model.eval()
            pred_val, _ = self.model.predict(self.val_set.values)
            val_Loss = self.model.loss(torch.Tensor(pred_val[:, 0]),
                                        torch.Tensor(self.val_label.astype('float32').values.reshape(-1, )))
            if val_Loss < best_val_loss:
                best_val_loss = val_Loss
                es = 0
            else:
                es += 1
            if es > 15:
                print(f'Early stopping with epoch{iter_count} val loss {best_val_loss}')
                break

            total_loss_list.append(total_loss_)

        self.model.eval()
        pred_tra, _ = self.model.predict(self.train_set.values)
        pred_val, _ = self.model.predict(self.val_set.values)
        pred_test, _ = self.model.predict(self.test_set.values)
        pred_tra_df = pd.DataFrame(pred_tra[:, 0], index=self.train_label.index, columns=[self.target['label1']])
        pred_val_df = pd.DataFrame(pred_val[:, 0], index=self.val_label.index, columns=[self.target['label1']])
        pred_test_df = pd.DataFrame(pred_test[:, 0], index=self.test_label.index, columns=[self.target['label1']])
        # imp_feat = self.imp_feat()
        return pred_tra_df, pred_val_df, pred_test_df, self.model

    def imp_feat(self):
        return None

    class MLP_base(nn.Module):
        def __init__(self, dim_input, protos, flag_nec, seed=2022):
            super(MLP.MLP_base, self).__init__()
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.dim_input = dim_input
            self.num_proto = 2
            self.num_classes = 2
            self.dim_list = [dim_input, max(dim_input // 1, 4), max(dim_input // 2, 4), max(dim_input // 3, 3), max(dim_input // 4, 2), self.num_classes]
            self.flag_nec = flag_nec
            if protos is not None:
                self.protos = torch.Tensor(protos.values)
                self.protos_label = torch.Tensor(protos.values)
            else:
                self.protos = nn.Parameter(
                    torch.randn(self.num_proto * self.num_classes, self.dim_list[-2]),
                    requires_grad=True
                )

            self.radius = nn.Parameter(
                torch.rand(1, self.num_proto * self.num_classes) - 0.6,
                requires_grad=True
            )
            self.fc1 = nn.Linear(self.dim_list[0], self.dim_list[1])
            self.fc2 = nn.Linear(self.dim_list[1], self.dim_list[2])
            self.fc3 = nn.Linear(self.dim_list[2], self.dim_list[4])
            self.fc4 = nn.Linear(self.dim_list[4], self.dim_list[4])
            self.linear = nn.Linear(self.dim_list[4], self.dim_list[-1])
            self.list_feat = [self.dim_list[1], self.dim_list[2], self.dim_list[4]]
            self.weight_init()

        def weight_init(self):
            for key_, block in self._modules.items():
                if key_ == 'loss_pred':
                    continue
                try:
                    for m in self._modules[key_]:
                        self.kaiming_init(m)
                except:
                    self.kaiming_init(block)

        def forward(self, x):
            out1 = self.fc1(x)
            out1_act = F.gelu(out1)
            out2 = self.fc2(out1_act)
            out2_act = F.gelu(out2)
            out3 = self.fc3(out2_act)
            out3_act = F.gelu(out3)
            out4 = self.fc4(out3_act)
            if self.flag_nec:
                prob = self.nce_prob_cos(out4)
                pass
            else:
                prob = self.linear(out4)
                prob = torch.sigmoid(prob)
                pass
            return prob, [out1, out2, out3, out4]

        def loss(self, x, y):
            loss_CE = nn.BCELoss()
            CE_loss = loss_CE(x, y)
            loss_ = CE_loss
            return loss_

        def predict(self, data):
            pred = self.forward(torch.Tensor(data))
            prob = pred[0].detach().numpy()
            feat1 = pred[1][0].detach().numpy()
            feat2 = pred[1][1].detach().numpy()
            feat3 = pred[1][2].detach().numpy()
            feat4 = pred[1][3].detach().numpy()
            return prob, [feat1, feat2, feat3, feat4]

        '''
        Reference:
        https://github.com/WanFang13/NCE-Net
        '''
        def nce_prob_cos(self, feat):
            dist = self.cosine_distance_func(feat, self.protos)
            dist = (dist / self.radius.sigmoid()).sigmoid()
            cls_score, _ = dist.view(-1, self.num_proto, self.num_classes).max(1)
            return cls_score

        def nce_prob_euc(self, feat):
            dist = self.euclidean_distance_func(feat.sigmoid(), self.protos.sigmoid())
            dist = torch.exp(-(dist ** 2) / (2 * self.radius.sigmoid() ** 2))
            cls_score, _ = dist.view(-1, self.num_proto, self.num_classes).max(1)
            return cls_score

        @staticmethod
        def kaiming_init(m):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)

        @staticmethod
        def normed_euclidean_distance_func(feat1, feat2):
            # Normalized Euclidean Distance
            # feat1: N * Dim
            # feat2: M * Dim
            # out:   N * M Euclidean Distance
            feat1, feat2 = F.normalize(feat1), F.normalize(feat2)
            feat_matmul = torch.matmul(feat1, feat2.t())
            distance = torch.ones_like(feat_matmul) - feat_matmul
            distance = distance * 2
            return distance.clamp(1e-10).sqrt()

        @staticmethod
        def euclidean_distance_func(feat1, feat2):
            # Euclidean Distance
            # feat1: N * Dim
            # feat2: M * Dim
            # out:   N * M Euclidean Distance
            feat1_square = torch.sum(torch.pow(feat1, 2), 1, keepdim=True)
            feat2_square = torch.sum(torch.pow(feat2, 2), 1, keepdim=True)
            feat_matmul = torch.matmul(feat1, feat2.t())
            distance = feat1_square + feat2_square.t() - 2 * feat_matmul
            return distance.clamp(1e-10).sqrt()

        @staticmethod
        def cosine_distance_func(feat1, feat2):
            # feat1: N * Dim
            # feat2: M * Dim
            # out:   N * M Cosine Distance
            distance = torch.matmul(F.normalize(feat1), F.normalize(feat2).t())
            return distance

        @staticmethod
        def cosine_distance_full_func(feat1, feat2):
            # feat1: N * Dim
            # feat2: M * Dim
            # out:   (N+M) * (N+M) Cosine Distance
            feat = torch.cat((feat1, feat2), dim=0)
            distance = torch.matmul(F.normalize(feat), F.normalize(feat).t())
            return distance

