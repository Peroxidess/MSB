import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from preprocess.get_dataset import MyDataset
from model.ae import AE


class ActiveLearning():
    def __init__(self, args, shape_inp, name_method_AL='ours'):
        self.seed = args.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        args.ae_shape_inp = shape_inp
        args.ae_latent_dim = max(shape_inp // 8, 6)
        args.ae_train_iterations = 150
        args.ae_beta = 0.01
        self.name_method = name_method_AL
        self.model_AL = BinsFuzz(args, name_method_AL)

    def preprocessing(self, train_x, val_x):
        self.model_AL.fit_transform_ae(train_x, val_x)

    def data_pool_init(self, train_x, label, method_init):
        label_data_index, unlabel_data_index = self.model_AL.data_pool_init(train_x, method_init, num_init=12)
        label_data = train_x.loc[label_data_index]
        unlabel_data = train_x.loc[unlabel_data_index]
        label_data_label = label.loc[label_data_index]
        unlabel_data_label = label.loc[unlabel_data_index]
        return label_data, label_data_label, unlabel_data, unlabel_data_label

    def data_choose(self, model, label_data, train_label, unlabel_data, unlabel_data_label, num_choose_AL, method_name_AL='ours', target={'label1': 'label'}, epoch_AL=0):
        pred_diff_index = self.model_AL.sample_choose(model, unlabel_data, unlabel_data_label, method_AL=method_name_AL,
                                                      num_choose=num_choose_AL, epoch_AL=epoch_AL)
        samples = unlabel_data.loc[pred_diff_index]
        samples_label = unlabel_data_label.loc[pred_diff_index][[target['label1']]]
        unlabel_data.drop(index=pred_diff_index, inplace=True)
        unlabel_data_label = unlabel_data_label.loc[unlabel_data.index]
        label_data = pd.concat([label_data, samples])
        label_data_label = pd.concat([train_label, samples_label])
        return label_data, label_data_label, unlabel_data, unlabel_data_label


class BinsFuzz():
    def __init__(self, args, name_method):
        self.args = args
        self.seed = self.args.seed
        self.args.cuda = args.cuda and torch.cuda.is_available()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.name_method = name_method
        self.ae = AE([args.ae_shape_inp, args.ae_shape_inp//2, args.ae_shape_inp//4], args.ae_latent_dim)

    def data_pool_init(self, data, method_init, num_init=4):
        data = pd.DataFrame(self.hidden_pred_ae(data), index=data.index)
        if 'ours' in method_init:
            if 'singlesmall' in method_init:
                bins_stride_list = [5, 13, 23, 37, 47, 61]
            elif 'singlelarge' in method_init:
                bins_stride_list = [5, 13, 23, 37, 47, 61]
            elif 'multi' in method_init:
                bins_stride_list = [5, 13, 23, 37, 47, 61]
            data_bins_sum_df = pd.DataFrame([])
            for bins_stride in bins_stride_list:
                data_bins_sum = pd.DataFrame(self.scale_margin(data, bins_stride), index=data.index)
                data_bins_sum_df = pd.concat([data_bins_sum_df, data_bins_sum], axis=1)
            data_bins_sum_sort = data_bins_sum_df.max(axis=1).sort_values(ascending=False)
            data_index_BinsNum_max = data_bins_sum_sort.iloc[:num_init//2].index
            data_index_BinsNum_min = data_bins_sum_sort.iloc[-num_init//2:].index
            data_index_BinsNum = np.hstack((data_index_BinsNum_min, data_index_BinsNum_max))
            data_label = data.loc[data_index_BinsNum]
        else:
            data_label = data.sample(n=num_init, random_state=self.seed)
        data_unlabel = data.drop(index=data_label.index)
        return data_label.index, data_unlabel.index

    def fit_transform_ae(self, dataset, dataset_val):
        data_labeled = dataset
        data_unlabeled = dataset
        data_labeled = MyDataset(data_labeled, None)
        data_unlabeled = MyDataset(data_unlabeled, None)
        optim_vae = optim.Adam(self.ae.parameters(), lr=1e-3, weight_decay=1e-4)
        if self.args.cuda:
            self.ae = self.ae.cuda()
        data_labeled_loader = Data.DataLoader(data_labeled, batch_size=256, worker_init_fn=np.random.seed(self.seed))
        data_unlabeled_loader = Data.DataLoader(data_unlabeled, batch_size=256, worker_init_fn=np.random.seed(self.seed))
        for iter_count in range(self.args.ae_train_iterations):
            total_vae_loss_ = 0
            for data_labeled_batch, data_unlabeled_batch in zip(data_labeled_loader, data_unlabeled_loader): #labeled data loader is the same as unlabeled data loader here
                if self.args.cuda:
                    data_labeled_batch = data_labeled_batch.cuda()
                self.ae.train()
                optim_vae.zero_grad()
                recon_labeled, z = self.ae(data_labeled_batch)
                rec_loss = self.ae.loss(data_labeled_batch, recon_labeled)
                total_vae_loss = rec_loss
                total_vae_loss.backward()
                optim_vae.step()
                total_vae_loss_ += total_vae_loss.detach().numpy()
            print(f'train loss {total_vae_loss_}')
            self.ae.eval()
            recon_labeled, z = self.ae(torch.Tensor(dataset_val.values))
            rec_loss = self.ae.loss(torch.Tensor(dataset_val.values), recon_labeled)
            total_vae_loss = rec_loss
            if iter_count % 20 == 0:
                print(f'val loss {total_vae_loss}')
        pass

    def hidden_pred_ae(self, data):
        if 'aenot' in self.name_method or data.shape[1] <= 2:
            return data
        else:
            data_torch = torch.Tensor(data.values)
            self.ae.eval()
            recon_labeled, z = self.ae(data_torch)
            z_ = z.detach().numpy()
            z_df = pd.DataFrame(z_, index=data.index)
            return z_df

    def scale_abnormal(self, data, bins_stride=3):
        data_col_bins_abnormal_sum = pd.Series(np.zeros(shape=(data.shape[0],)), data.index)
        for col_index, values_ in data.iteritems():
            data_col = values_.values
            set_data_col = set(data_col)
            bins = max(int(np.floor(len(set_data_col) / bins_stride)), 2)
            histogram_data_col = np.histogram(data_col, bins=bins)
            index_hist_num_threshold = copy.deepcopy(histogram_data_col[0])
            index_hist_num_threshold[index_hist_num_threshold < index_hist_num_threshold.max() * 0.1] = 0
            index_hist_num = copy.deepcopy(index_hist_num_threshold)
            histogram_data_col_IndexNonzero = pd.Series(np.where(index_hist_num_threshold != 0)[0])
            histogram_data_col_IndexNonzero_pre = histogram_data_col_IndexNonzero.shift(1)
            histogram_data_col_IndexNonzero_post = histogram_data_col_IndexNonzero.shift(-1)
            histogram_data_col_IndexNonzero_pre.fillna(-1, inplace=True)
            histogram_data_col_IndexNonzero_post.fillna(index_hist_num_threshold.shape[0], inplace=True)
            indexzero_pre = histogram_data_col_IndexNonzero - histogram_data_col_IndexNonzero_pre - 1
            indexzero_post = histogram_data_col_IndexNonzero_post - histogram_data_col_IndexNonzero - 1
            indexzero_sum = indexzero_pre + indexzero_post
            index_hist_num[np.where(index_hist_num_threshold != 0)[0]] = indexzero_sum
            abnormal_ratio = index_hist_num / bins
            indexzero_label = abnormal_ratio

            data_col_bins_abnormal = pd.cut(data_col, bins=bins, labels=indexzero_label, ordered=False)
            data_col_bins_abnormal = pd.Series(data_col_bins_abnormal, index=data.index).astype('float32')
            data_col_bins_abnormal_sum += data_col_bins_abnormal / data.shape[1]
        data_col_bins_abnormal_sum = pd.Series(data_col_bins_abnormal_sum, index=data.index)

        return data_col_bins_abnormal_sum

    def scale_margin(self, data, bins_stride=3):
        data_bins_sum = None
        for col_index, values_ in data.iteritems():
            data_col = values_.values
            set_data_col = set(data_col)
            bins = max(int(np.floor(len(set_data_col) / bins_stride)), 2)
            histogram_data_col = np.histogram(data_col, bins=bins)
            data_col_bins_num = pd.cut(data_col, bins=bins, labels=histogram_data_col[0], ordered=False).astype('int32')
            if data_bins_sum is not None:
                data_bins_sum += data_col_bins_num
            else:
                data_bins_sum = copy.deepcopy(data_col_bins_num)
        data_bins_sum_nor = data_bins_sum / bins_stride
        return data_bins_sum_nor

    def scale_fuzz(self, data, bins_stride=3):
        data_fuzz = None
        for col_index, values_ in data.iteritems():
            data_col = values_.values
            set_data_col = set(data_col)
            bins = max(int(np.floor(len(set_data_col) / bins_stride)), 2)

            histogram_data_col = np.histogram(data_col, bins=bins)

            histogram_per = pd.Series(histogram_data_col[1])
            histogram_post = pd.Series(histogram_data_col[1]).shift()
            histogram_ave_ = (histogram_per + histogram_post)/2
            histogram_ave_ = histogram_ave_.dropna(axis=0, how='any')

            data_col_bins = pd.cut(data_col, bins=bins, labels=histogram_ave_, ordered=False)
            data_col_bins = pd.Series(data_col_bins)
            if data_fuzz is None:
                data_fuzz = data_col_bins
            else:
                data_fuzz = np.vstack((data_fuzz, data_col_bins))
        data_fuzz = np.transpose(data_fuzz)
        data_fuzz_df = pd.DataFrame(data_fuzz, index=data.index, columns=data.columns)
        return data_fuzz_df

    def sample_choose(self, model, unlabel_data, unlabel_data_label, method_AL='ours', num_choose=80, epoch_AL=0):
        hidden_z_tra_unlabel = self.hidden_pred_ae(unlabel_data)
        if 'ours' in method_AL:
            try:
                pred_train = model.predict_proba(hidden_z_tra_unlabel)
            except:
                pred_train, _ = model.predict(hidden_z_tra_unlabel.values)
            if 'singlesmall' in method_AL:
                bins_stride_list = [5, 13, 23, 37, 47, 61]
            elif 'singlelarge' in method_AL:
                bins_stride_list = [5, 13, 23, 37, 47, 61]
            elif 'multi' in method_AL:
                bins_stride_list = [5, 13, 23, 37, 47, 61]
            pred_diff_ratio_df_all = pd.DataFrame([])
            for bins_stride in bins_stride_list:
                hidden_z_tra_unlabel_fuzz = self.scale_fuzz(hidden_z_tra_unlabel, bins_stride)
                hidden_z_tra_unlabel_abnormal = self.scale_abnormal(hidden_z_tra_unlabel, bins_stride)
                try:
                    pred_train_fuzz = model.predict_proba(hidden_z_tra_unlabel_fuzz)
                except:
                    pred_train_fuzz, _ = model.predict(hidden_z_tra_unlabel_fuzz.values)

                pred_confusion = (1 + pred_train - pred_train.max(axis=1).reshape(-1, 1)).sum(axis=1).reshape(-1, )
                pred_confusion_fuzz = (1 + pred_train_fuzz - pred_train_fuzz.max(axis=1).reshape(-1, 1)).sum(axis=1).reshape(-1, )
                if 'nec' in method_AL:
                    pred_diff = pred_confusion - pred_confusion_fuzz
                else:
                    pred_diff_ = pred_train - pred_train_fuzz
                    pred_diff = pred_diff_[:, 0]
                pred_diff_ratio = pred_diff / pred_train[:, 0]
                pred_diff_df = pd.Series(abs(pred_diff), index=unlabel_data.index)
                pred_diff_ratio_df = pd.Series(abs(pred_diff_ratio), index=unlabel_data.index)
                data_bins_sum = pd.Series(self.scale_margin(unlabel_data, bins_stride), index=unlabel_data.index)
                if 'nondiversity' in method_AL:
                    initial_criteria = pred_diff_df
                    Intermediate_criteria = pred_diff_df
                    final_criteria = pred_diff_ratio_df
                elif 'nonuncertainty' in method_AL:
                    initial_criteria =hidden_z_tra_unlabel_abnormal + 1e-2 * (data_bins_sum / data_bins_sum.max())
                    Intermediate_criteria = hidden_z_tra_unlabel_abnormal + 1e-2 * (data_bins_sum / data_bins_sum.max())
                    final_criteria = hidden_z_tra_unlabel_abnormal + 1e-2 * (data_bins_sum / data_bins_sum.max())
                else:
                    initial_criteria = hidden_z_tra_unlabel_abnormal + 1e-2 * (data_bins_sum / data_bins_sum.max())
                    Intermediate_criteria = pred_diff_df
                    final_criteria = pred_diff_ratio_df
                alpha = 1 / (epoch_AL + 1) ** 5
                beta = 1 - alpha
                pred_diff = alpha * initial_criteria + beta * Intermediate_criteria + 1e-3 * final_criteria
                pred_diff_ratio_df_all = pd.concat([pred_diff_ratio_df_all, pred_diff], axis=1)
            pred_diff_ratio_sort = pred_diff_ratio_df_all.max(axis=1).sort_values(ascending=False)
            index_ = pred_diff_ratio_sort.index[:num_choose]
            pred_diff_index = index_
            unlabel_data_label_plot = copy.deepcopy(unlabel_data_label)
            unlabel_data_label_plot.loc[pred_diff_index] = 2
        return pred_diff_index
