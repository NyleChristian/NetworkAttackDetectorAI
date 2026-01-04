import pandas as pd
import sklearn as sk
import os
from dotenv import load_dotenv 
from common import *
from preprocessingfunctions import *
from sklearn.preprocessing
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
#df['labels'] = df['labels'].map(lambda x: 0 if x == 'normal' else 1)

attackTypeMap = {
    ATTACK_TYPE_DOS : 1,
    ATTACK_TYPE_PROBE : 2,
    ATTACK_TYPE_R2L : 3,
    ATTACK_TYPE_U2R : 4,
    NORMAL : 0 
}


#group attacks by type
#create new column attack_type from labels with constant attack type names
#create new column attack_type_number from labels with 0-4 values


df['attack_type'] = df['labels'].map(divByAttack)
#df['attack_type_number'] = df['attack_type'].map(attackTypeMap)




#1-hot encode attack_type column 
# add new binary columns for each unique value in attack_type
df = OneHotEncColumn(df, 'attack_type')

#drop original labels column
df.drop('labels', axis=1, inplace=True)
#1-hot encode protocol_type, service, flag columns
df = OneHotEncColumn(df, 'protocol_type')
df = OneHotEncColumn(df, 'service')
df = OneHotEncColumn(df, 'flag')

#drops these original columns as model only needs numeric data
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

# print("dos",dos_df.shape)
# print("probe",probe_df.shape)
# print("r2l",r2l_df.shape)
# print("u2r",u2r_df.shape)


# print(dos_df['src_bytes'])

# scaler_dos = StandardScaler().fit(dos_df)
# dos_df = scaler_dos.transform(dos_df)
# scaler_probe = StandardScaler().fit(probe_df)
# probe_df = scaler_probe.transform(probe_df)
# scaler_r2l = StandardScaler().fit(r2l_df)
# r2l_df = scaler_r2l.transform(r2l_df)
# scaler_u2r = StandardScaler().fit(u2r_df)
# u2r_df = scaler_u2r.transform(u2r_df)

#===============================Random Forest for feature importance on Dos dataset
#move dos_attack flag to end
dos_df = dos_df.drop(['attack_type_PROBE','attack_type_R2L','attack_type_U2R', 'attack_type_NORMAL'], axis=1)
dos_df = moveColumnToEnd(dos_df, 'attack_type_DOS')
dos_X = dos_df.iloc[:, :-1]
dos_y = dos_df.iloc[:, -1]

dos_x_train, dos_x_test, dos_y_train, dos_y_test = train_test_split(dos_X, dos_y, test_size=0.2, random_state=42)



classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(dos_x_train, dos_y_train)
dos_y_pred = classifier.predict(dos_x_test)

dos_accuracy = accuracy_score(dos_y_test, dos_y_pred)

print(f'DOS Dataset Accuracy: {dos_accuracy * 100 :.2f}')
print(f'Recall: {recall_score(dos_y_test, dos_y_pred):.2f}')
print(f'F1 Score: {f1_score(dos_y_test, dos_y_pred):.2f}')

print("\nClassification Report:\n", classification_report(dos_y_test,dos_y_pred))


# dos_conf_matrix = confusion_matrix(dos_y_test, dos_y_pred)

# plt.figure(figsize=(8, 6))
# sns.heatmap(dos_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Normal', 'DOS Attack'], yticklabels=['Normal', 'DOS Attack'])
# plt.title('DOS Attack Confusion Matrix')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.show()

dos_feature_importances = classifier.feature_importances_
dos_feature_names = classifier.feature_names_in_
print(f'DOS Feature Importances: {dos_feature_importances}')

dos_feature_importance_df = pd.DataFrame({'Feature': dos_feature_names, 'Importance': dos_feature_importances})
dos_feature_importance_df = dos_feature_importance_df.sort_values(by='Importance', ascending=False)
dos_feature_importance_df = dos_feature_importance_df.head(20)



plt.barh(dos_feature_importance_df['Feature'], dos_feature_importance_df['Importance'] )
plt.xlabel("Feature Importance")
plt.title("DOS Feature Importance in Random Forest Classifier")
plt.show()







#print(dos_df.head)





# print('Test set:')
# for col_name in df.columns:
#     if df[col_name].dtype == 'object':
#         unique_values = len(df[col_name].unique())
#         print(f'Column: {col_name}, Unique Values: {unique_values}')


#print(df.head())

