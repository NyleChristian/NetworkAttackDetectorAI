from common import *
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt


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

#def getInitTrainingAccuracy(col1, col2, col3, col4)

def getNImportantFeatures(df, n = 20, attack_type = 'DOS', confusion : bool = False, graph: bool = False):
    
    current_attack_type = f'attack_type_{attack_type}' 
    df = moveColumnToEnd(df, current_attack_type)
    df = df.rename(columns={current_attack_type : attack_type})
    df = df.drop(df.filter(regex='attack_type_').columns, axis = 1, inplace = True)

    df_X = df.iloc[:, :-1]
    df_y = df.iloc[:, -1]

    df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)



    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(df_x_train, df_y_train)
    df_y_pred = classifier.predict(df_x_test)

    df_accuracy = accuracy_score(df_y_test, df_y_pred)


    print(f'{attack_type} Dataset Accuracy: {df_accuracy * 100 :.2f}')
    print(f'Recall: {recall_score(df_y_test, df_y_pred):.2f}')
    print(f'F1 Score: {f1_score(df_y_test, df_y_pred):.2f}')

    print("\nClassification Report:\n", classification_report(df_y_test,df_y_pred))

    if confusion == True:
        df_conf_matrix = confusion_matrix(df_y_test, df_y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(df_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Normal', f'{attack_type} Attack'], yticklabels=['Normal', f'{attack_type} Attack'])
        plt.title('DOS Attack Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    df_feature_importances = classifier.feature_importances_
    df_feature_names = classifier.feature_names_in_

    feature_importance_df = pd.DataFrame({'Feature': df_feature_names, 'Importance': df_feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=True)
    feature_importance_df = feature_importance_df.head(n)

    if graph == True:
        
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'] )
        plt.xlabel("Feature Importance")
        plt.title("DOS Feature Importance in Random Forest Classifier")
        plt.show()



