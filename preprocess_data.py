#Prepare Dataset Feature set i
import glob
import pandas as pd
import scipy.io as sio
import numpy as np

seed = 42

def data_preprocess_csv(files, data_type):
    """
    input: multiple mean_p_area.csv files
    output: a dataframe used for classification analysis
    This function will select alpha power ratio and low gamma power ratio 
    and it will transform multiple mean_p_area.csv to a new dataframe with classification death or not
    """
    #standardized the path name, get the subject who both has asleep and awake status
    if data_type == "original":
        raw_subjects = [subject[26:-22] for subject in files]
    elif data_type == "generated":
        raw_subjects = [subject[12:-22] for subject in files]
    
    subjects = []
    for subject in raw_subjects:
        if subject[-1] != "_":
            subject += "_"
        subjects.append(subject)
    subjects = set(subjects)
    
    #create path
    directory = "original_data_mean_p_area/" # mean_p_area/
    asleep = "asleep_mean_p_area.csv"
    awake = "awake_mean_p_area.csv"
    
    #initiate empty df
    columns = ["subject"] + ["alpha_power_ratio_" + "g" + str(i) for i in range(1,10)] + \
    ["low_gamma_power_ratio_" + "g" + str(i) for i in range(1,10)] + \
    ["death"]
    res_df = pd.DataFrame(columns=columns)
    
    #calculating alpha_power ratio and low_gamma_power_ratio
    for subject in subjects:
        path_asleep =  directory + subject + asleep
        df_asleep = pd.read_csv(path_asleep, index_col=0).loc[:,['alpha', 'low_gamma']]
        
        path_awake =  directory + subject + awake
        df_awake = pd.read_csv(path_awake, index_col=0).loc[:,['alpha', 'low_gamma']]
        
        alpha_power_ratio = (df_asleep['alpha'] / df_awake['alpha']).tolist() #g1 to g9
        low_gamma_power_ratio = (df_asleep['low_gamma'] / df_awake['low_gamma']).tolist() #g1 to g9
        
        if "C1" in subject or "C2" in subject:
            death = [0]
        else:
            death = [1]
        
        if data_type == "generated":
            new_row = ["generated"+subject] + alpha_power_ratio + low_gamma_power_ratio + death
        elif data_type == "original":
            new_row = [subject] + alpha_power_ratio + low_gamma_power_ratio + death
            
        res_df.loc[len(res_df)] = new_row
    return res_df


def data_preprocess_mat(file = "mean_p_area.mat"):
    """
    input: file, path to the mat file contains mean_p_area
    output: res_df, a dataframe used for classification analysis
    This function will select alpha power ratio and low gamma power ratio 
    and it will transform multiple mean_p_area.mat to a new dataframe with classification death or not
    """
    #load the mat file
    df = sio.loadmat(file)['mean_p_area'][1:] #this gives an array
    
    #individual column name for each patient
    columns = ["high_gamma", "low_gamma", "beta", "alpha", "theta", "delta"]
    
    #initiate the returned df
    res_columns = ["subject"] + ["alpha_power_ratio_" + "g" + str(i) for i in range(1,10)] + \
    ["low_gamma_power_ratio_" + "g" + str(i) for i in range(1,10)] + \
    ["death"]
    res_df = pd.DataFrame(columns=res_columns)
    
    for i in range(len(df)):
        if df[i][1].size > 0 and df[i][2].size > 0:
            #ith patient, 1 = asleep status, 2 = awake status, (0 = row name,...)
            df_asleep = pd.DataFrame(df[i][1], columns = columns) 
            df_awake = pd.DataFrame(df[i][2], columns = columns)
            res_df.loc[len(res_df)] = [df[i][0][0]] + (df_asleep['alpha'] / df_awake['alpha']).tolist() + \
            (df_asleep['low_gamma'] / df_awake['low_gamma']).tolist() + \
            [1]
            
        if df[i][3].size > 0 and df[i][4].size > 0:
            df_asleep_c1 = pd.DataFrame(df[i][3], columns = columns)
            df_awake_c1 = pd.DataFrame(df[i][4], columns = columns)
            res_df.loc[len(res_df)] = [df[i][0][0]+"c1"] + (df_asleep_c1['alpha'] / df_awake_c1['alpha']).tolist() + \
            (df_asleep_c1['low_gamma'] / df_awake_c1['low_gamma']).tolist() + \
            [0]
            
        if df[i][5].size > 0 and df[i][6].size > 0:
            df_asleep_c2 = pd.DataFrame(df[i][5], columns = columns)
            df_awake_c2 = pd.DataFrame(df[i][6], columns = columns)
            res_df.loc[len(res_df)] = [df[i][0][0]+"c2"] + (df_asleep_c2['alpha'] / df_awake_c2['alpha']).tolist() + \
            (df_asleep_c2['low_gamma'] / df_awake_c2['low_gamma']).tolist() + \
            [0]
    
    
    return res_df

def add_hrv_helper(res_df_dead, hrv_df_dead):
    lfnu_list = []
    hfnu_list = []
    sd1_list = []
    ratio_sd2_sd1_list = []
    
    hrv_df_asleep = pd.DataFrame()
    hrv_df_awake = pd.DataFrame()
    
    for i in range(len(res_df_dead)):
        hrv_df_asleep = hrv_df_dead[hrv_df_dead['Unnamed: 0'].str.contains(res_df_dead.iloc[i]['subject'], case=False) \
                             & hrv_df_dead['Unnamed: 0'].str.contains('asleep', case=False)]
        hrv_df_awake = hrv_df_dead[hrv_df_dead['Unnamed: 0'].str.contains(res_df_dead.iloc[i]['subject'], case=False) \
                             & hrv_df_dead['Unnamed: 0'].str.contains('awake', case=False)]

        if hrv_df_asleep.empty or hrv_df_awake.empty:
            lfnu_list.append(np.nan)
            hfnu_list.append(np.nan)
            sd1_list.append(np.nan)
            ratio_sd2_sd1_list.append(np.nan)
        else:
            lfnu_list.append(hrv_df_asleep.iloc[0]['lfnu'] / hrv_df_awake.iloc[0]['lfnu'])
            hfnu_list.append(hrv_df_asleep.iloc[0]['hfnu'] / hrv_df_awake.iloc[0]['hfnu'])
            sd1_list.append(hrv_df_asleep.iloc[0]['sd1'] / hrv_df_awake.iloc[0]['sd1'])
            ratio_sd2_sd1_list.append(hrv_df_asleep.iloc[0]['ratio_sd2_sd1'] / hrv_df_awake.iloc[0]['ratio_sd2_sd1'])
        
        hrv_df_asleep = pd.DataFrame()
        hrv_df_awake = pd.DataFrame()
    
    res_df_dead.loc[:, 'lfnu'] = lfnu_list
    res_df_dead.loc[:, 'hfnu'] = hfnu_list
    res_df_dead.loc[:, 'sd1'] = sd1_list
    res_df_dead.loc[:, 'ratio_sd2_sd1'] = ratio_sd2_sd1_list
    
    return res_df_dead
    
def add_hrv_original(res_df, file = "mean_p_area.mat"):
    
    hrv_df_c1 = pd.read_excel('Control_1.xlsx')[['Unnamed: 0', "lfnu", "hfnu", "sd1", "ratio_sd2_sd1"]]
    hrv_df_c2 = pd.read_excel('Control_2.xlsx')[['Unnamed: 0', "lfnu", "hfnu", "sd1", "ratio_sd2_sd1"]]
    hrv_df_dead = pd.read_excel('SUDEP.xlsx')[['Unnamed: 0', "lfnu", "hfnu", "sd1", "ratio_sd2_sd1"]]
    
    res_df_dead = res_df[res_df['death'] == 1]
    res_df_c1 = res_df[res_df['subject'].str.contains("c1")]
    res_df_c2 = res_df[res_df['subject'].str.contains("c2")]

    return pd.concat([add_hrv_helper(res_df_dead, hrv_df_dead),
                      add_hrv_helper(res_df_c1, hrv_df_c1),
                      add_hrv_helper(res_df_c2, hrv_df_c2)])

def generate_hrv_features(data, num_bootstrap_samples):
    bootstrap_means = []

    for _ in range(num_bootstrap_samples):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)
    return bootstrap_means

def add_hrv(original_df, generated_df):
    generated_df_dead = generated_df[generated_df['death'] == 1]
    generated_df_living = generated_df[generated_df['death'] == 0]
    
    original_df_dead = original_df[original_df['death'] == 1]
    original_df_living = original_df[original_df['death'] == 0]
    
    num_bootstrap_samples = len(generated_df_dead)
    for colname in ["lfnu", "hfnu", "sd1", "ratio_sd2_sd1"]:
        generated_df_dead.loc[:, colname] = generate_hrv_features(original_df_dead[colname].tolist(), num_bootstrap_samples)
    
    num_bootstrap_samples = len(generated_df_living)
    for colname in ["lfnu", "hfnu", "sd1", "ratio_sd2_sd1"]:
        generated_df_living.loc[:, colname] = generate_hrv_features(original_df_living[colname].tolist(), num_bootstrap_samples)
    
    return pd.concat([generated_df_dead, generated_df_living, original_df])