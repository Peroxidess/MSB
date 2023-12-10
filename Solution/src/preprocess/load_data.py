import numpy as np
import pandas as pd


def data_load(task_name):
    test_data = pd.DataFrame([])

    if 'thyroid' in task_name:
        file_name_tra = '../DataSet/UCI_thyroid/ann-train.csv'
        file_name_test = '../DataSet/UCI_thyroid/ann-test.csv'
        data = pd.read_csv(file_name_tra, sep=' ', header=None).dropna(how='all', axis=1)
        data_test = pd.read_csv(file_name_test, sep=' ', header=None).dropna(how='all', axis=1)
        train_data = data.rename(columns={data.shape[1] - 1: 'label1'})
        test_data = data_test.rename(columns={data.shape[1] - 1: 'label1'})
        train_data['label1'][train_data['label1'] == 3] = 0
        test_data['label1'][test_data['label1'] == 3] = 0
        train_data['label1'][(train_data['label1'] == 2) | (train_data['label1'] == 1)] = 1
        test_data['label1'][(test_data['label1'] == 2) | (test_data['label1'] == 1)] = 1
        target_dict = {'label1': 'label1'}
    elif 'arrhythmia' in task_name:
        file_name_tra = '../DataSet/UCI_arrhythmia/arrhythmia.csv'
        data = pd.read_csv(file_name_tra, header=None).dropna(how='all', axis=1)
        train_data = data.rename(columns={data.shape[1] - 1: 'label1'})
        unknow_index = train_data['label1'][train_data['label1'] == 16].index
        train_data.drop(index=unknow_index, inplace=True)
        train_data['label1'][train_data['label1'] != 1] = 0
        train_data[train_data == '?'] = np.nan
        target_dict = {'label1': 'label1'}
    elif 'breast' in task_name:
        file_name_tra = '../DataSet/UCI_breast_cancer/breast-cancer-wisconsin.csv'
        data = pd.read_csv(file_name_tra, header=None, index_col=[0]).dropna(how='all', axis=1)
        data.reset_index(drop=True, inplace=True)
        data_drop_dup = data.drop_duplicates()
        train_data = data_drop_dup.rename(columns={data_drop_dup.shape[1]: 'label1'})
        train_data['label1'][train_data['label1'] == 2] = 0
        train_data['label1'][(train_data['label1'] == 4)] = 1
        target_dict = {'label1': 'label1'}
    elif 'heart' in task_name:
        path_root = '../DataSet/UCI_heart_disease/'
        path_file = 'yanxishe.csv'
        target_dict = {'label1': 'target'}
        data = pd.read_csv(path_root+path_file, index_col=['id'])
        train_data = data

    elif 'mimic' in task_name:
        if 'ppc' in task_name:
            file_name = '../DataSet/mimic/data_preprocessed_row.csv'
            data = pd.read_csv(file_name, index_col=['subject_id'])
            target_dict = {'label1': 'label1'}
            train_data = data.rename(columns={'label_dead': 'label1'})
    return train_data, test_data, target_dict
