from common import *
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np 



def divByAttack(label):
    match label:
        #CASE 1: DOS
        case 'neptune' | 'smurf' | 'apache2' | 'teardrop' | 'back' | 'pod' | 'processtable' | 'mailbomb' | 'land': 
            return ATTACK_TYPE_DOS
        #CASE 2: PROBE
        case 'portsweep' | 'satan' | 'ipsweep' | 'nmap' | 'mscan' | 'saint':
            return ATTACK_TYPE_PROBE
        #CASE 3: R2L
        case 'guess_passwd' | 'warezclient' | 'warezmaster' | 'httptunnel' | 'snmpguess' | 'snmpgetattack' | 'multihop' | 'xsnoop' | 'imap' | 'sendmail' | 'phf' | 'xlock' | 'ftp_write' | 'named':
            return ATTACK_TYPE_R2L
        #CASE 4: U2R
        case 'buffer_overflow' | 'ps' | 'rootkit' | 'xterm' | 'loadmodule' | 'perl':
            return ATTACK_TYPE_U2R
        #CASE 0: NORMAL
        case 'normal' | 'norma':
            return NORMAL
        case _:
            print("Unknown label:", label)
            return NORMAL

def OneHotEncColumn(df, col_name):
# Initialize the encoder and fit/transform the specified column
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[[col_name]])

    encoded_cols = encoder.get_feature_names_out([col_name])
    df_encoded = pd.DataFrame(encoded_data, columns=encoded_cols)
    
  
    df_final = pd.concat([df, df_encoded], axis=1)
    return df_final

def moveColumnToEnd(df, col_name):
    cols = list(df.columns)
    cols.remove(col_name)
    cols.append(col_name)
    df = df[cols]
    return df
