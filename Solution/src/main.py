import copy
import time
import pandas as pd
import argparse
import arguments
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from preprocess import load_data
from preprocess.get_dataset import DataPreprocessing
from preprocess.missing_values_imputation import MVI
from model.evaluate import Eval_Regre, Eval_Class
from model.baseline import Baseline, MLP
from model.ActiveLearning import ActiveLearning

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def model_tra_eval(train_set, train_label, val_set, val_label, test_set, test_label,
                   target, co_col, ca_col, task_name, nor_std, param_init, seed):
    train_set: pd.DataFrame
    train_label: pd.DataFrame
    val_set: pd.DataFrame
    val_label: pd.DataFrame
    test_set: pd.DataFrame
    val_label: pd.DataFrame
    target: dict

    metric_all = pd.DataFrame([])
    imp_feat_ = pd.DataFrame([])
    name_model = 'MLP_Base'
    model_method = MLP(name_model,
                       train_set, train_label,
                       val_set, val_label,
                       test_set, test_label, target, co_col, ca_col, task_name, seed,
                       param_init,
                       param_fit={'epoch': 300}
                       )
    pred_tra, pred_val, pred_test, model = model_method.grid_fit_pred(batch_size=min(train_set.shape[0], 64))
    if 'class' in task_name:
        metric = Eval_Class
    else:
        metric = Eval_Regre
    for index_, values in pred_tra.iteritems():
        metric_tra = metric(train_label.loc[:, index_].values.reshape(-1, 1),
                            pred_tra.loc[:, index_].values.reshape(-1, 1), train_set,
                            'train', task_name, name_model[1], nor_flag=False, nor_std=nor_std).eval()
        metric_val = metric(val_label.loc[:, index_].values.reshape(-1, 1),
                            pred_val.loc[:, index_].values.reshape(-1, 1), val_set,
                            'val', task_name, name_model[1], nor_flag=False, nor_std=nor_std).eval()
        metric_test = metric(test_label.loc[:, index_].values.reshape(-1, 1),
                             pred_test.loc[:, index_].values.reshape(-1, 1), test_set,
                             'test', task_name, name_model[1], nor_flag=False, nor_std=nor_std).eval()
        metric_single = pd.concat([metric_test, metric_val, metric_tra], axis=1)
        metric_all = pd.concat([metric_all, metric_single], axis=0)
    return metric_all, pd.DataFrame(pred_tra, columns=train_label.columns, index=train_label.index), \
               pd.DataFrame(pred_test, columns=test_label.columns, index=test_label.index), imp_feat_, model


def run(train_data, test_data, target, args, trial) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    target: dict
    if args.test_ratio == 0 or not test_data.empty:
        train_set = train_data
        test_set = test_data
    else:
        train_set, test_set = train_test_split(train_data, test_size=args.test_ratio, random_state=args.seed)

    metric_df_all = pd.DataFrame([])
    pred_train_df_all = pd.DataFrame([])
    pred_test_df_all = pd.DataFrame([])
    metric_AL_AllFlod = pd.DataFrame([])
    kf = KFold(n_splits=args.n_splits)
    for k, (train_index, val_index) in enumerate(kf.split(train_set)):
        print(f'KFlod {k}')
        metric_all_fold = pd.DataFrame([])
        train_set_cv = train_set.iloc[train_index]
        val_set_cv = train_set.iloc[val_index]
        test_set_cv = copy.deepcopy(test_set)

        dp = DataPreprocessing(train_set_cv, val_set_cv, test_set_cv, target, seed=args.seed,
                               flag_label_onehot=False,
                               flag_ex_null=True, flag_ex_std_flag=False, flag_ex_occ=False,
                               flag_ca_co_sel=True, flag_ca_fac=True, flag_onehot=True, flag_nor=True,
                               flag_feat_emb=False, flag_RUS=False, flag_confusion=False, flaq_save=False)
        if args.Flag_DataPreprocessing:
            train_set_cv, val_set_cv, test_set_cv, ca_col, co_col, nor = dp.process()

        col_drop = dp.features_ex(train_set_cv,) # Drop useless features (high deletion rate, small variance, etc.)
        train_set_cv.drop(columns=col_drop, inplace=True)
        col_ = train_set_cv.columns
        val_set_cv = val_set_cv[col_]
        test_set_cv = test_set_cv[col_]
        ca_col = train_set_cv.filter(regex=r'sparse').columns.tolist()
        co_col = train_set_cv.filter(regex=r'dense').columns.tolist()

        train_label = train_set_cv[[target['label1']]]
        val_label = val_set_cv[[target['label1']]]
        test_label = test_set_cv[[target['label1']]]
        train_x = train_set_cv.drop(columns=target.values())
        val_x = val_set_cv.drop(columns=target.values())
        test_x = test_set_cv.drop(columns=target.values())

        print(f'train_x shape {train_x.shape} | val_x shape {val_x.shape} | test_x shape {test_x.shape}')

        # missing values imputation start
        if args.Flag_MVI:
            mvi = MVI(co_col, ca_col, args.seed)
            train_x_filled = mvi.fit_transform(train_x)
            val_x_filled = mvi.transform(val_x)
            test_x_filled = mvi.transform(test_x)
        # missing values imputation end

        # active learning start
        if args.method_AL is not None and 'nec_' in args.method_AL:
            flag_nec = True
        else:
            flag_nec = False
        param_init = {'protos': None, 'flag_nec': flag_nec}
        if args.method_AL is not None:
            al = ActiveLearning(args, train_x_filled.shape[1], args.method_AL)
            al.preprocessing(train_x_filled, val_x_filled)
            label_data, label_data_label, unlabel_data, unlabel_data_label = al.data_pool_init(train_x_filled, train_label, args.method_AL)

            metric_AL_iter = pd.DataFrame([])
            metric = pd.DataFrame([])

            args.num_choose_AL = 4
            for epoch_AL in range(80):

                hidden_z_tra = al.model_AL.hidden_pred_ae(label_data)
                hidden_z_tra_unlabel = al.model_AL.hidden_pred_ae(unlabel_data)
                hidden_z_val = al.model_AL.hidden_pred_ae(val_x_filled)
                hidden_z_test = al.model_AL.hidden_pred_ae(test_x_filled)

                metric, pred_train_df, pred_test_df, imp_feat, model = model_tra_eval(hidden_z_tra, label_data_label,
                                                                               hidden_z_val, val_label,
                                                                               hidden_z_test, test_label,
                                                                               target, co_col, ca_col, args.task_name, nor, param_init,
                                                                                      args.seed)

                metric_AL_iter = pd.concat([metric_AL_iter, metric])

                label_data, label_data_label, unlabel_data, unlabel_data_label = al.data_choose(model, label_data, label_data_label,
                                                                                                unlabel_data, unlabel_data_label,
                                                                                                num_choose_AL=args.num_choose_AL,
                                                                                                method_name_AL=args.method_AL,
                                                                                                target=target,
                                                                                                epoch_AL=epoch_AL)

                if unlabel_data.empty or unlabel_data.shape[0] < args.num_choose_AL:
                    break
            metric_AL_iter.columns = pd.MultiIndex.from_product([[f'flod {10 * trial + k}'], metric_AL_iter.columns])
            metric_AL_AllFlod = pd.concat([metric_AL_AllFlod, metric_AL_iter], axis=1)

        metric.index = [args.method_mvi]
        metric_all_fold = pd.concat([metric_all_fold, metric], axis=1)
        metric_df_all = pd.concat([metric_df_all, metric_all_fold], axis=0)
        metric_AL_AllFlod.to_csv(f'./{args.task_name}_metric_AL_{args.method_AL}_flod{k}.csv')
    return metric_df_all, pred_train_df_all, pred_test_df_all, metric_AL_AllFlod


if __name__ == "__main__":
    args = arguments.get_args()

    test_prediction_all = pd.DataFrame([])
    train_prediction_all = pd.DataFrame([])
    history_df_all = pd.DataFrame([])
    metric_df_all = pd.DataFrame([])
    metric_AL_Allrun = pd.DataFrame([])

    for trial in range(args.nrun):
        print('rnum : {}'.format(trial))
        args.seed = (trial * 55) % 2022 + 1 # a different random seed for each run

        # data fetch
        # input: file path
        # output: data with DataFrame
        train_data, test_data, target = load_data.data_load(args.task_name)

        # run model
        # input: train_data
        # output: metric, train_prediction, test_prediction
        metric_df, train_prediction, test_prediction, metric_AL_AllFlod = run(train_data, test_data, target, args, trial)

        metric_df_all = pd.concat([metric_df_all, metric_df], axis=0)
        test_prediction_all = pd.concat([test_prediction_all, test_prediction], axis=1)
        train_prediction_all = pd.concat([train_prediction_all, train_prediction], axis=1)
        metric_AL_Allrun = pd.concat([metric_AL_Allrun, metric_AL_AllFlod], axis=1)

        local_time = time.strftime("%m_%d_%H_%M", time.localtime())
        metric_AL_Allrun.to_csv(f'./{args.task_name}_metric_AL_{args.method_AL}_trial{trial}.csv')
    metric_df_all.to_csv(f'./{args.task_name}_{local_time}.csv', index_label=['index'])

    # print metric
    metric_df_all['model'] = metric_df_all.index
    metric_mean = metric_df_all.groupby('model').mean()
    metric_mean_test = metric_mean.filter(regex=r'test')
    metric_mean_val = metric_mean.filter(regex=r'val')
    print(metric_mean)
    print('mean test auc_: ', metric_mean_test['test auc_'])
    print('mean val auc_: ', metric_mean_val['val auc_'])
    pass
pass
