import pandas as pd
import sklearn as sk
import os
from dotenv import load_dotenv 
from common import *
from functions import *
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np

load_dotenv()

#Open dataset
path = os.getenv('INPUT_FILE')
df = pd.read_csv(path)
df = df.drop_duplicates()
print("Initial shape:", df.shape)
print(df.columns[df.isnull().any()].tolist())

print(df['labels'].unique)



#Simplifies 'labels' column into either normal or one of 4 attack type categories to make future encoding easier
#Originally, there were 36 different categories under 'labels' column
df['attack_type'] = df['labels'].map(divByAttack)

#Adds 5 new columns with purely binary entries, one each for NORMAL and the aforementioned 4 attack types
df = OneHotEncColumn(df, 'attack_type')

#drops nonnumerical categorical columns as model works best with numerical values
df.drop('labels', axis=1, inplace=True)
#1-hot encode protocol_type, service, flag columns
df = OneHotEncColumn(df, 'protocol_type')
df = OneHotEncColumn(df, 'service')
df = OneHotEncColumn(df, 'flag')

#drops these original catgegorical columns once encoded, as model only needs numeric data
df = df.drop(['protocol_type', 'service', 'flag', 'attack_type'], axis=1)

#drop rows with NaN values
df = df.dropna()

#check shape after preprocessing
print("Shape after preprocessing:", df.shape)

#=========================split dataset into 4 subsets based on attack type, including normal traffic in each subset
dos_df = df[(df['attack_type_DOS'] == 1) | (df['attack_type_NORMAL'] == 1)]
probe_df = df[(df['attack_type_PROBE'] == 1) | (df['attack_type_NORMAL'] == 1)]
r2l_df = df[(df['attack_type_R2L'] == 1) | (df['attack_type_NORMAL'] == 1)]
u2r_df = df[(df['attack_type_U2R'] == 1) | (df['attack_type_NORMAL'] == 1)]




df_topfeatures = getNImportantFeatures(dos_df, 20, ATTACK_TYPE_DOS)
trainModel(df_topfeatures, ATTACK_TYPE_DOS, False)


df_topfeatures = getNImportantFeatures(probe_df, 20, ATTACK_TYPE_PROBE)
trainModel(df_topfeatures, ATTACK_TYPE_PROBE, False)

df_topfeatures = getNImportantFeatures(r2l_df, 20, ATTACK_TYPE_R2L, True)
trainModelForLowOccurrence(df_topfeatures, ATTACK_TYPE_R2L, False)

df_topfeatures = getNImportantFeatures(u2r_df, 20, ATTACK_TYPE_U2R,True, False, True)
trainModelForLowOccurrence(df_topfeatures, ATTACK_TYPE_U2R, False)
#print(dos_df.head)





# print('Test set:')
# for col_name in df.columns:
#     if df[col_name].dtype == 'object':
#         unique_values = len(df[col_name].unique())
#         print(f'Column: {col_name}, Unique Values: {unique_values}')


#print(df.head())

