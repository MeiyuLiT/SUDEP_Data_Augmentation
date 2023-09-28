import scipy.io
import pandas as pd
import numpy as np
from scipy import stats
def mat_to_csv_complexity_features():
    file_paths = ['ha_complexity_ratios_simulate.mat', 'dfa_complexity_ratios_simulate.mat', 'kol_complexity_ratios_simulate.mat']
    # file_paths = ['ha_complexity_ratios.mat', 'dfa_complexity_ratios.mat', 'kol_complexity_ratios.mat']
    res_dfs = []
    for file_path in file_paths:
        if 'ha_' in file_path:
            column_names = ['subject'] + [f'ha_g{i}' for i in range(1,10)]
        elif 'dfa_' in file_path:
            column_names = ['subject'] + [f'dfa_g{i}' for i in range(1,10)]
        elif 'kol_' in file_path:
            column_names = ['subject'] + [f'kol_g{i}' for i in range(1,10)]

        mat = scipy.io.loadmat('complexity_features/' + file_path)['complexity_ratios'][0][0]
        complexity_df = []
        for i in range(len(mat)):
            mat_i = mat[i].flatten()
            new_row = [value[0] for value in mat_i]
            res_row = [new_row[0]] + [value[0] for value in new_row[1:]]
            complexity_df.append(res_row)

        single_df = pd.DataFrame(complexity_df, columns = column_names)
        res_dfs.append(single_df)
    res_df = pd.merge(res_dfs[0], res_dfs[1], on="subject")
    res_df = pd.merge(res_df, res_dfs[2], on="subject")
    return 0
df = pd.read_csv('complexity_features.csv', index_col=0)
df = df.fillna(0)
df = df.drop(df[df.iloc[:,1:].eq(0).all(axis=1)].index)

for col in df.columns[1:]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    threshold = 20
    outliers = df[(df[col] < Q1 - threshold * IQR) | (df[col] > Q3 + threshold * IQR)]
    df = df.drop(outliers.index.tolist())
    print(len(outliers))
df.to_csv('processed_complexity_features.csv')