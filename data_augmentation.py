import os
from glob import glob

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

from fooof import FOOOF
from scipy.signal import welch
from scipy.integrate import simps, simpson
import scipy
from fooof.sim.gen import gen_aperiodic

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True, bidirectional = True)
        self.linear = nn.Linear(50*2, 1)#50*1 for bidirectional = False
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    def predict(self, input_data):
        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            output_tensor = self(input_tensor)
            self.train()  # Set the model back to training mode
            return output_tensor.squeeze(0).numpy() 

        
def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

def get_simulated_data(timeseries, lookback = 5, n_epochs = 1):
    # train-test split for time series
    train_size = int(len(timeseries) * 0.6)
    test_size = len(timeseries) - train_size
    train, test = timeseries[:train_size], timeseries[train_size:]
    
    #create dataset
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)
    
    #create model
    model = AirModel()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

    #run epochs
    test_rmse_list = np.array([])
    for epoch in range(0, n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 2 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        test_rmse_list = np.append(test_rmse_list, test_rmse)
        #print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(timeseries) * np.nan
        y_pred = model(X_train)
        y_pred = y_pred[:, -1, :]
        train_plot[lookback:train_size] = model(X_train)[:, -1, :]
        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan
        yhats = model(X_test)[:, -1, :]
        test_plot[train_size+lookback:len(timeseries)] = yhats
        
    return np.concatenate(np.array(yhats))

#Calculate mean_p_area
def calculate_mean_p_area(simulated_data, fs, low, high):
    fm = FOOOF()

    (f, s) = scipy.signal.welch(simulated_data, fs, scaling = 'spectrum')

    freq_range = [3, 100]#?????
    fm.fit(f, s, freq_range)

    init_ap_fit = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
    init_flat_spec = fm.power_spectrum - init_ap_fit
    init_flat_spec = 10 ** init_flat_spec
    total_power = simpson(init_flat_spec, fm.freqs)

    idx = np.logical_and(low <= fm.freqs, fm.freqs <= high)
    absolute_power = simpson(init_flat_spec[idx], fm.freqs[idx])
    mean_p_area = absolute_power / total_power
    return mean_p_area

def simulate_and_mean_p_area_dict(df):

# 1. simulate data
# 2. give the mean_p_area for each column, categorized by 6 bands

    fs = (df['sample'][len(df)-1] - df['sample'][0]) // (df['time'][len(df)-1] - df['time'][0])

    #bands
    delta = (0.5, 4)
    theta = (4, 8)
    alpha = (8, 12)
    beta = (12, 30)
    low_gamma = (30, 50)
    high_gamma = (50, 100)
    bands = (high_gamma, low_gamma, beta, alpha, theta, delta)

    mean_p_area_dict = {}

    all_simulated_data = {}
    for column in df.columns:
        #choose the EEG data to simulate, delete columns sample, time
        if 'sample' not in column and 'time' not in column and 'ekg' not in column:
            timeseries_list = df[[column]].values.astype('float32')

            #divide into n chuncks
            n = 1000
            timeseries_list = [timeseries_list[i * n:(i + 1) * n] for i in range((len(timeseries_list) + n - 1) // n )] 

            #start simulating
            simulated_data = []
            for timeseries in timeseries_list:
                if len(timeseries) == n:
                    simulated_data = np.append(simulated_data, get_simulated_data(timeseries, lookback = 5, n_epochs = 5))
            
            all_simulated_data[column] = simulated_data
            
            #calculate mean_p_area
            mean_p_area_bands = []
            for band in bands:
                low = band[0]
                high = band[1]
                mean_p_area = calculate_mean_p_area(simulated_data, fs, low, high)
                mean_p_area_bands.append(mean_p_area)
            mean_p_area_dict[column] = np.array(mean_p_area_bands)

            
#         elif 'ekg' in column:
#             n = 1000
#             timeseries_list = [timeseries_list[i * n:(i + 1) * n] for i in range((len(timeseries_list) + n - 1) // n )] 
#             print("Start processing ", column, ": ", len(timeseries_list), " lists with size ", n)

#             #start simulating
#             simulated_data_ekg = []
#             for timeseries in timeseries_list:
#                 simulated_data_ekg = np.append(simulated_data_ekg, get_simulated_data(timeseries, lookback = 5, n_epochs = 5))
    
    return mean_p_area_dict, all_simulated_data

#get the mean_p_area_df for each file, average the mean among group
def mean_p_area_df(mean_p_area_dict):
    g1 = g2 = g3 = g4 = g5 = g6 = g7 = g8 = g9 = np.zeros(6) #6 bands
    g1_count, g2_count, g3_count, g4_count, g5_count, g6_count, g7_count, g8_count, g9_count = 0,0,0,0,0,0,0,0,0
    for key in mean_p_area_dict.keys():
        if 'fp1' in key or 'f7' in key or 'f9' in key:
            g1 = np.add(g1, mean_p_area_dict[key])
            g1_count += 1
        elif 'fp2' in key or 'f8' in key or 'f10' in key:
            g2 = np.add(g2, mean_p_area_dict[key])
            g2_count += 1
        elif 'f3' in key or 'f4' in key or 'fz' in key:
            g3 = np.add(g3, mean_p_area_dict[key])
            g3_count += 1
        elif 't9' in key or 't7' in key or 'a1' in key:
            g4 = np.add(g4, mean_p_area_dict[key])
            g4_count += 1
        elif 'c3' in key or 'cz' in key or 'c4' in key:
            g5 = np.add(g5, mean_p_area_dict[key])
            g5_count += 1
        elif 't8' in key or 't10' in key or 'a2' in key:
            g6 = np.add(g6, mean_p_area_dict[key])
            g6_count += 1
        elif 'p9' in key or 'p7' in key or 'o1' in key:
            g7 = np.add(g7, mean_p_area_dict[key])
            g7_count += 1
        elif 'p3' in key or 'pz' in key or 'p4' in key:
            g8 = np.add(g8, mean_p_area_dict[key])
            g8_count += 1
        elif 'o2' in key or 'p8' in key or 'p10' in key:
            g9 = np.add(g9, mean_p_area_dict[key])
            g9_count += 1

    g1 = np.divide(g1 , g1_count)
    g2 = np.divide(g2 , g2_count)
    g3 = np.divide(g3 , g3_count)
    g4 = np.divide(g4 , g4_count)
    g5 = np.divide(g5 , g5_count)
    g6 = np.divide(g6 , g6_count)
    g7 = np.divide(g7 , g7_count)
    g8 = np.divide(g8 , g8_count)
    g9 = np.divide(g9 , g9_count)

    print(g1_count, g2_count, g3_count, g4_count, g5_count, g6_count, g7_count, g8_count, g9_count)

    mean_ap_area_df = pd.DataFrame([g1, g2, g3, g4, g5, g6, g7, g8, g9], 
                                   columns = ["high_gamma", "low_gamma", "beta", "alpha", "theta", "delta"])

    return mean_ap_area_df

if __name__ == "__main__":
    print("#################Get File Names#################")
    directory = {'COL': 'EEG files - Zhe Chen/COLUMBIA UNIVERSITY EEG/',
                'JHU': 'EEG files - Zhe Chen/JOHNS HOPKINS UNIVERSITY EEG/',
                'Austin': 'EEG files - Zhe Chen/MELBOURNE EEG/AUSTIN EEG/',
                'NYU': 'EEG files - Zhe Chen/NYU EEG/',
                "Yale": 'EEG files - Zhe Chen/YALE UNIVERSITY EEG/',
                "CIN": "EEG files - Zhe Chen/UNIVERSITY OF CINCINNATI EEG/",
                "RMH": 'EEG files - Zhe Chen/MELBOURNE EEG/RMH EEG/',
                "STV": 'EEG files - Zhe Chen/MELBOURNE EEG/ST. VINCENT EEG/'}  


    file_pattern_asleep = {"COL": ['COL SUDEP 1 Asleep.txt', 'COL SUDEP 2 Asleep.txt', 'COL SUDEP 3 Asleep.txt', 'COL SUDEP 4 Asleep.txt', 'COL SUDEP 5 Asleep.txt',  'COL SUDEP 7 Asleep.txt'],#没有6
                            "JHU": ['JH SUDEP1 Asleep 7.5min.txt'],
                            "Austin": ["AUSTIN S1 asleep.txt", "AUSTIN S6 asleep.txt", "AUSTIN S9 asleep.txt", "AUSTIN S10 asleep.txt", "AUSTIN S11 asleep (8.5min).txt", "AUSTIN S12 asleep (6.5min).txt"],
                            "NYU": ["NYU SUDEP 1 Asleep.txt", "NYU SUDEP 2 Asleep.txt"],
                        "Yale": ['YALE-SUDEP1 asleep.txt','YALE-SUDEP2 asleep.txt','YALE-SUDEP3 asleep.txt','YALE-SUDEP4 asleep.txt','YALE-SUDEP5 asleep.txt'],
                        "CIN": ["CIN SUDEP 1 Asleep 6min.txt", "CIN SUDEP 2 Asleep.txt"],
                        "RMH": ["RMH-SUDEP1 asleep.txt", "RMH-SUDEP4 asleep.txt", "RMH-SUDEP6 asleep.txt", "RMH-SUDEP11 asleep.txt", "RMH-SUDEP15 asleep.txt"],
                        "STV": ["STV_SUDEP5 asleep 8min.txt", "STV_SUDEP9 asleep 6min.txt"]
                        }


    file_pattern_awake = {"COL": ['COL SUDEP 1 Awake.txt', 'COL SUDEP 2 Awake.txt', 'COL SUDEP 3 Awake.txt', 'COL SUDEP 4 Awake.txt', 'COL SUDEP 5 Awake.txt', 'COL SUDEP 6 Awake 6min.txt', 'COL SUDEP 7 Awake 5.5min.txt'],
                        "JHU": ['JH SUDEP1 Awake.txt'],
                        "Austin": ["AUSTIN S1 awake.txt", "AUSTIN S6 awake.txt", "AUSTIN S9 awake.txt", "AUSTIN S10 awake (5.5min).txt", "AUSTIN S11 awake.txt", "AUSTIN S12 awake (6min).txt"],
                        "NYU": ["NYU SUDEP 1 Awake.txt", "NYU SUDEP 2 Awake.txt"],
                        "Yale": ['YALE-SUDEP1 awake.txt', 'YALE-SUDEP2 awake.txt', 'YALE-SUDEP3 awake.txt', 'YALE-SUDEP4 awake.txt', 'YALE-SUDEP5 awake.txt'],
                        "CIN": ["CIN SUDEP 1 awake.txt", "CIN SUDEP 2 awake 7min.txt"],
                        "RMH": ["RMH-SUDEP1 awake.txt", "RMH-SUDEP4 awake.txt", "RMH-SUDEP6 awake.txt"],#没有11，15
                        "STV": ["STV_SUDEP5 awake.txt"]#没有9
                        
                        }

                                
    files_asleep = []
    files_awake = []
    for key, direct in directory.items():
        
        for f_asleep in range(len(file_pattern_asleep[key])):
            files_asleep.append(direct + file_pattern_asleep[key][f_asleep])
        for f_awake in range(len(file_pattern_awake[key])):
            files_awake.append(direct + file_pattern_awake[key][f_awake])

    c_file_pattern_asleep = {"COL": [
    'COL SUDEP 1C1 Asleep (7min).txt',
    'COL SUDEP 1C2 Asleep.txt',
    'COL SUDEP 2C1 Asleep 6min .txt',
    'COL SUDEP 2C2 Asleep 7min.txt',
    'COL SUDEP 3C1 Asleep.txt',
    'COL SUDEP 3C2 Asleep.txt',
    'COL SUDEP 4C1 Asleep.txt',
    'COL SUDEP 4C2 Asleep.txt',
    'COL SUDEP 5C1 Asleep.txt',
    'COL SUDEP 6C1 Asleep 5min.txt',
    'COL SUDEP 6C2 Asleep 8min.txt',
    'COL SUDEP 7C1 Asleep 5min.txt',
    'COL SUDEP 7C2 Asleep 6min.txt'
    ],
    "JHU": ['JH SUDEP1C1 Asleep 6min.txt', 'JH SUDEP1C2 Asleep 5min.txt'],
    "Austin": [
    'AUSTIN S1C1 asleep.txt',
    'AUSTIN S1C2 asleep.txt',
    'AUSTIN S6C1 asleep (6.5min).txt',
    'AUSTIN S6C2 asleep (7.5min).txt',
    'AUSTIN S9C1 asleep.txt',
    'AUSTIN S9C2 asleep.txt',
    'AUSTIN S10C1 asleep (7min).txt',
    'AUSTIN S10C2 asleep.txt',
    'AUSTIN S11C1 asleep (5.5min).txt',
    'AUSTIN S11C2 asleep.txt',
    'AUSTIN S12C1 asleep.txt',
    'AUSTIN S12C2 asleep (6.5min).txt'
    ],
    "NYU": [
    "NYU SUDEP 1C1 Asleep.txt",
    "NYU SUDEP 1C2 Asleep.txt",
    "NYU SUDEP 2C1 Asleep.txt",
    "NYU SUDEP 2C2 Asleep.txt"],
    "Yale": [
    'YALE-SUDEP1C1 asleep.txt',
    'YALE-SUDEP1C2 asleep.txt',
    'YALE-SUDEP2C1 asleep.txt',
    'YALE-SUDEP2C2 asleep.txt',
    'YALE-SUDEP3C1 asleep.txt',
    'YALE-SUDEP3C2 asleep.txt',
    'YALE-SUDEP4C1 asleep.txt',
    'YALE-SUDEP4C2 asleep.txt',
    'YALE-SUDEP5C2 asleep.txt'#没有5c1
    ],
    "CIN": [
    'CIN SUDEP 1C1 asleep 8min.txt',
    'CIN SUDEP 2C1 asleep 7min.txt',#没有1c2
    'CIN SUDEP 2C2 asleep 8min.txt'
    ],
    "RMH": [
    'RMH-SUDEP1C1 asleep.txt',
    'RMH-SUDEP1C2 asleep.txt',
    'RMH-SUDEP4C1 asleep.txt',
    'RMH-SUDEP4C2 asleep.txt',
    'RMH-SUDEP6C1 asleep 4min.txt',#没有6c2，没有11c1, 11c2
    'RMH-SUDEP15C1 asleep.txt',
    'RMH-SUDEP15C2 asleep.txt'
    ],
    "STV": [
    'STV_SUDEP5C1 asleep.txt',
    'STV_SUDEP5C2 asleep.txt',
    'STV_SUDEP9C1 asleep.txt',
    'STV_SUDEP9C2 asleep.txt'
    ]
                        }


    c_file_pattern_awake = {"COL": [
    'COL SUDEP 1C1 Awake.txt',
    'COL SUDEP 1C2 Awake.txt',
    'COL SUDEP 2C1 Awake.txt',
    'COL SUDEP 2C2 Awake 7min.txt',
    'COL SUDEP 3C1 Awake.txt',
    'COL SUDEP 3C2 Awake.txt',
    'COL SUDEP 4C1 Awake.txt',
    'COL SUDEP 4C2 Awake.txt',
    'COL SUDEP 5C1 Awake.txt',
    'COL SUDEP 5C2 Awake 8min.txt',
    'COL SUDEP 6C1 Awake.txt',
    'COL SUDEP 6C2 Awake.txt',
    'COL SUDEP 7C1 Awake 7min.txt',
    'COL SUDEP 7C2 Awake.txt'
    ],
    "JHU": ['JH SUDEP1C1 Awake 6min.txt', 'JH SUDEP1C2 Awake 8min.txt'],
    "Austin": [
    'AUSTIN S1C1 awake.txt',
    'AUSTIN S1C2 awake.txt',
    'AUSTIN S6C1 awake (7min).txt',
    'AUSTIN S6C2 awake.txt',
    'AUSTIN S9C1 awake.txt',
    'AUSTIN S9C2 awake (6min).txt',
    'AUSTIN S10C1 awake (8.5min).txt',
    'AUSTIN S10C2 awake (6.5min).txt',
    'AUSTIN S11C1 awake (5min).txt',
    'AUSTIN S11C2 awake (6.5min).txt',
    'AUSTIN S12C1 awake.txt',
    'AUSTIN S12C2 awake.txt'
    ],
    "NYU": [
    "NYU SUDEP 1C1 Awake.txt",
    "NYU SUDEP 1C2 Awake  6min.txt",
    "NYU SUDEP 2C1 Awake.txt",
    "NYU SUDEP 2C2 Awake.txt"],
    "Yale": [
    'YALE-SUDEP1C1 awake.txt',
    'YALE-SUDEP1C2 awake.txt',
    'YALE-SUDEP2C1 awake.txt',
    'YALE-SUDEP2C2 awake.txt',
    'YALE-SUDEP3C1 awake.txt',
    'YALE-SUDEP3C2 awake.txt',
    'YALE-SUDEP4C1 awake.txt',
    'YALE-SUDEP4C2 awake.txt',
    'YALE-SUDEP5C1 awake.txt',
    'YALE-SUDEP5C2 awake.txt'
    ],
    "CIN": [
    'CIN SUDEP 1C1 awake 7min.txt',
    'CIN SUDEP 1C2 awake.txt',
    'CIN SUDEP 2C1 awake 6min.txt',
    'CIN SUDEP 2C2 awake 6.5min.txt'
    ],
    "RMH": [
    'RMH-SUDEP1C1 awake.txt',
    'RMH-SUDEP1C2 awake.txt',
    'RMH-SUDEP4C1 awake.txt',
    'RMH-SUDEP4C2 awake.txt',
    'RMH-SUDEP6C1 awake.txt',#没有6c2，11c2
    'RMH-SUDEP11C1 awake.txt',
    'RMH-SUDEP15C2 awake.txt'
    ],
    "STV": ["STV_SUDEP9C1 awake.txt"]#没有9c2
    }

    c1_files_asleep = []
    c1_files_awake = []
    c2_files_asleep = []
    c2_files_awake = []
    for key, direct in directory.items():
        
        for f_asleep in range(len(c_file_pattern_asleep[key])):
            if "C1" in c_file_pattern_asleep[key][f_asleep]:
                c1_files_asleep.append(direct + c_file_pattern_asleep[key][f_asleep])
            else:
                c2_files_asleep.append(direct + c_file_pattern_asleep[key][f_asleep])
        for f_awake in range(len(c_file_pattern_awake[key])):
            if "C1" in c_file_pattern_awake[key][f_awake]:
                c1_files_awake.append(direct + c_file_pattern_awake[key][f_awake])
            else:
                c2_files_awake.append(direct + c_file_pattern_awake[key][f_awake])

    print("The number of patients: ")
    print("Asleep dead: ", len(files_asleep))
    print("Awake dead: ", len(files_awake))
    print("Asleep living c1: ", len(c1_files_asleep))
    print("Awake living c1: ", len(c1_files_awake))
    print("Asleep living c2: ", len(c2_files_asleep))
    print("Awake living c2: ", len(c2_files_awake))

    print("#################Read Each txt File#################")

    files_asleep_data = []
    for i in range(len(files_asleep)):
        temp_df = pd.read_csv(files_asleep[i], delimiter="\t", header = 1)
        temp_df.columns = [x.lower().strip() for x in temp_df.columns]
        files_asleep_data.append(temp_df)
        
    files_awake_data = []
    for i in range(len(files_awake)):
        temp_df = pd.read_csv(files_awake[i], delimiter="\t", header = 1)
        temp_df.columns = [x.lower().strip() for x in temp_df.columns]
        files_awake_data.append(temp_df)
        
    c1_files_asleep_data = []
    for i in range(len(c1_files_asleep)):
        temp_df = pd.read_csv(c1_files_asleep[i], delimiter="\t", header = 1)
        temp_df.columns = [x.lower().strip() for x in temp_df.columns]
        c1_files_asleep_data.append(temp_df)
        
    c1_files_awake_data = []
    for i in range(len(c1_files_awake)):
        temp_df = pd.read_csv(c1_files_awake[i], delimiter="\t", header = 1)
        temp_df.columns = [x.lower().strip() for x in temp_df.columns]
        c1_files_awake_data.append(temp_df)
        
        
    c2_files_asleep_data = []
    for i in range(len(c2_files_asleep)):
        temp_df = pd.read_csv(c2_files_asleep[i], delimiter="\t", header = 1)
        temp_df.columns = [x.lower().strip() for x in temp_df.columns]
        c2_files_asleep_data.append(temp_df)
        
    c2_files_awake_data = []
    for i in range(len(c2_files_awake)):
        temp_df = pd.read_csv(c2_files_awake[i], delimiter="\t", header = 1)
        temp_df.columns = [x.lower().strip() for x in temp_df.columns]
        c2_files_awake_data.append(temp_df)

    
    print("#################Simulating Data, mean_p_area, and save#################")

    delta = (0.5, 4)
    theta = (4, 8)
    alpha = (8, 12)
    beta = (12, 30)
    low_gamma = (30, 50)
    high_gamma = (50, 100)
    bands = (high_gamma, low_gamma, beta, alpha, theta, delta)

    subjects = ('AUSTIN S9',
    'RMH-SUDEP15',
    'AUSTIN S6',
    'CIN SUDEP 2',
    'RMH-SUDEP6',
    'RMH-SUDEP1',
    'RMH-SUDEP4',
    'YALE-SUDEP2',
    'YALE-SUDEP3',
    'NYU SUDEP 2',
    'STV_SUDEP9',
    'YALE-SUDEP4',
    'AUSTIN S10',
    'YALE-SUDEP1',
    'YALE-SUDEP5',
    'COL SUDEP 2',
    'COL SUDEP 5',
    'JH SUDEP1',
    'AUSTIN S11',
    'AUSTIN S12',
    'CIN SUDEP 1',
    'NYU SUDEP 1',
    'AUSTIN S1',
    'STV_SUDEP5',
    'RMH-SUDEP11',
    'COL SUDEP 4',
    'COL SUDEP 6',
    'COL SUDEP 7',
    'COL SUDEP 1',
    'COL SUDEP 3')

    #'asleep', 'awake', 'C1_asleep', 'C1_awake', 'C2_asleep', 'C2_awake'
    #files_asleep, files_awake, c1_files_asleep, c1_files_awake, c2_files_asleep, c2_files_awake
    files_dict = {'asleep': files_asleep, 
                'awake': files_awake,
                'C1_asleep': c1_files_asleep,
                'C1_awake': c1_files_awake,
                'C2_asleep': c2_files_asleep,
                'C2_awake': c2_files_awake}

    directory_path_simulate_data = "./simulate_data/"
    directory_path_mean_p_area = "./mean_p_area/"
    mat_mean_p_area = {}
    for subject in tqdm(subjects):
        subject_mean_p_area = []
        for key in files_dict.keys():
            file_idx = 0
            eeg_mean_p_area_df = None
            for file_name in files_dict[key]:
                if subject in file_name and key == 'asleep':#if file exists
                    print(subject, "_", key, " is simulating and calculating mean_p_area")
                    df = files_asleep_data[file_idx]
                    mean_p_area_dict, all_simulated_data = simulate_and_mean_p_area_dict(df)
                    eeg_mean_p_area_df = mean_p_area_df(mean_p_area_dict)
                    all_simulated_data = pd.DataFrame.from_dict(all_simulated_data)
                    
                    break

                elif subject in file_name and key == 'awake':#if file exists
                    print(subject, "_", key, " is simulating and calculating mean_p_area")
                    df = files_awake_data[file_idx]
                    mean_p_area_dict, all_simulated_data = simulate_and_mean_p_area_dict(df)
                    eeg_mean_p_area_df = mean_p_area_df(mean_p_area_dict)
                    all_simulated_data = pd.DataFrame.from_dict(all_simulated_data)

                    break
                    
                elif subject in file_name and key == 'C1_asleep':#if file exists
                    print(subject, "_", key, " is simulating and calculating mean_p_area")
                    df = c1_files_asleep_data[file_idx]
                    mean_p_area_dict, all_simulated_data = simulate_and_mean_p_area_dict(df)
                    eeg_mean_p_area_df = mean_p_area_df(mean_p_area_dict)
                    all_simulated_data = pd.DataFrame.from_dict(all_simulated_data)

                    break
                    
                elif subject in file_name and key == 'C1_awake':#if file exists
                    print(subject, "_", key, " is simulating and calculating mean_p_area")
                    df = c1_files_awake_data[file_idx]
                    mean_p_area_dict, all_simulated_data = simulate_and_mean_p_area_dict(df)
                    eeg_mean_p_area_df = mean_p_area_df(mean_p_area_dict)
                    all_simulated_data = pd.DataFrame.from_dict(all_simulated_data)

                    break
                    
                elif subject in file_name and key == 'C2_asleep':#if file exists
                    print(subject, "_", key, " is simulating and calculating mean_p_area")
                    df = c2_files_asleep_data[file_idx]
                    mean_p_area_dict, all_simulated_data = simulate_and_mean_p_area_dict(df)
                    eeg_mean_p_area_df = mean_p_area_df(mean_p_area_dict)
                    all_simulated_data = pd.DataFrame.from_dict(all_simulated_data)

                    break
                    
                elif subject in file_name and key == 'C2_awake':#if file exists
                    print(subject, "_", key, " is simulating and calculating mean_p_area")
                    df = c2_files_awake_data[file_idx]
                    mean_p_area_dict, all_simulated_data = simulate_and_mean_p_area_dict(df)
                    eeg_mean_p_area_df = mean_p_area_df(mean_p_area_dict)
                    all_simulated_data = pd.DataFrame.from_dict(all_simulated_data)

                    break
                    
                file_idx += 1
            if eeg_mean_p_area_df is None:
                print(subject, "_" , key , " has no simulate data and no mean_p_area")
            else:
                eeg_mean_p_area_df.to_csv(directory_path_mean_p_area + subject + "_" + key + "_" + "mean_p_area.csv")
                all_simulated_data.to_csv(directory_path_simulate_data + subject + "_" + key + "_" + "all_simulated_data.csv") 


