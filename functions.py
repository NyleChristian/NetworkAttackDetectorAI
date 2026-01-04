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
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split


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


def getNImportantFeatures(df, n = 20, attack_type = 'DOS', useLowOccurenceModel : bool  = False, confusion : bool = False, graph: bool = False):
    
    current_attack_type = f'attack_type_{attack_type}' 
    df = moveColumnToEnd(df, current_attack_type)
    df = df.rename(columns={current_attack_type : attack_type})
    df = df.loc[:, ~df.columns.str.startswith('attack_type')]

    df_x_test, df_y_test, classifier = trainModelForLowOccurrence(df, attack_type, True) if useLowOccurenceModel else trainModel(df, attack_type,  True)

    df_y_pred = classifier.predict(df_x_test)

    df_accuracy = accuracy_score(df_y_test, df_y_pred)


    print(f'{attack_type} Dataset Accuracy: {df_accuracy * 100 :.2f}')
    print(f'Recall: {recall_score(df_y_test, df_y_pred):.2f}')
    print(f'F1 Score: {f1_score(df_y_test, df_y_pred):.2f}')

    #print(f"\nClassification Report{attack_type}:\n", classification_report(df_y_test,df_y_pred))

    if confusion == True:
        confusionMatrix(attack_type, df_y_test, df_y_pred)

    df_feature_importances = classifier.feature_importances_
    df_feature_names = classifier.feature_names_in_

    
    df_feature_names = classifier.feature_names_in_

    feature_importance_df = pd.DataFrame({'Feature': df_feature_names, 'Importance': df_feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
    feature_importance_df = feature_importance_df.head(n)

   
    if graph == True:
        graphFeatures(attack_type, feature_importance_df)

    return df [feature_importance_df['Feature'].tolist() + [attack_type]]


def trainModel(df, attack_type : str, hypertuning= False):

    df_X = df.iloc[:, :-1]
    df_y = df.iloc[:, -1]

    df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    df_x_train_scaled = scaler.fit_transform(df_x_train)
    df_x_test_scaled = scaler.fit_transform(df_x_test)
    
  
    #classifier = KNeighborsClassifier(n_neighbors = 2)
    #classifier = GaussianNB()
    classifier = RandomForestClassifier(n_estimators = 100, random_state=42)
    
    if hypertuning == True: 
        classifier = hypertune(df_x_train, df_y_train,classifier)
    else:
        classifier.fit(df_x_train, df_y_train)
        df_y_pred = classifier.predict(df_x_test)

        print(f"\nClassification Report Training - {attack_type}:\n", classification_report(df_y_test,df_y_pred))
        confusionMatrix(attack_type, df_y_test, df_y_pred)

    return df_x_test,df_y_test,classifier

def trainModelForLowOccurrence(df, attack_type: str, hypertuning = False):

    df_X = df.iloc[:, :-1]
    df_y = df.iloc[:, -1]

    df_x_train, _, df_y_train, _ = train_test_split(df_X, df_y, test_size=0.2, random_state=42)
    
    df_test_U2R = df[(df[attack_type] == 1)]
    df_test_normal = df[(df[attack_type] == 0)]
    df_test_normal = df_test_normal.head(len(df_test_U2R) * 2)
    df_test_combined = pd.concat([df_test_normal, df_test_U2R], ignore_index=True)
    
    df_combined_X = df_test_combined.iloc[:, :-1]
    df_combined_y = df_test_combined.iloc[:, -1]
    
    _, df_x_test, _, df_y_test = train_test_split(df_combined_X, df_combined_y, test_size=0.60, random_state=42)

    # df_x_train = df_X 
    # df_y_train = df_y
    # df_x_test  = df_combined_X
    # df_y_train = df_combined_y

    scaler = StandardScaler()
    df_x_train_scaled = scaler.fit_transform(df_x_train)
    df_x_test_scaled = scaler.fit_transform(df_x_test)
    
  
    #classifier = KNeighborsClassifier(n_neighbors = 2)
    #classifier = GaussianNB()
   
    classifier = RandomForestClassifier(random_state=42)
    
    if hypertuning == True: 
        classifier = hypertune(df_x_train, df_y_train,classifier)
    else:
        classifier.fit(df_x_train, df_y_train)
        df_y_pred = classifier.predict(df_x_test)

        print(f"\nClassification Report Training - {attack_type}:\n", classification_report(df_y_test,df_y_pred))
        confusionMatrix(attack_type, df_y_test, df_y_pred)

    return df_x_test,df_y_test,classifier



def trainModelKNeighbor(df, hypertuning):

    df_X = df.iloc[:, :-1]
    df_y = df.iloc[:, -1]

    df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    df_x_train_scaled = scaler.fit_transform(df_x_train)
    df_x_test_scaled = scaler.fit_transform(df_x_test)
    
  
    classifier = KNeighborsClassifier(n_neighbors = 2)
    #classifier = GaussianNB()
    #classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    
    if hypertuning == True: 
        classifier = hypertune(df_x_train, df_y_train,classifier)
    else:
        classifier.fit(df_x_train, df_y_train)
        df_y_pred = classifier.predict(df_x_test)

        print(f"\nClassification Report Training:\n", classification_report(df_y_test,df_y_pred))

    return df_x_test,df_y_test,classifier
def graphFeatures(attack_type, feature_importance_df):
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'] )
    plt.xlabel("Feature Importance")
    plt.title(f"{attack_type} Feature Importance in Random Forest Classifier")
    plt.show()

def confusionMatrix(attack_type, df_y_test, df_y_pred):
    df_conf_matrix = confusion_matrix(df_y_test, df_y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Normal', f'{attack_type} Attack'], yticklabels=['Normal', f'{attack_type} Attack'])
    plt.title(f'{attack_type} Attack Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

param_grid = {
    'n_estimators': [25],
    'max_features': ['sqrt'],
     'max_depth': [25],
     'min_samples_split': [2,5],
     'min_samples_leaf': [ 2, 4],
     'bootstrap': [True, False]
    }

def hypertune(df_x_train, df_y_train, classifier):
  
    grid_search = GridSearchCV(classifier, param_grid=param_grid, 
                            cv=3, n_jobs=-1, verbose=0, scoring='accuracy')
    grid_search.fit(df_x_train, df_y_train)
        
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score (accuracy): {grid_search.best_score_}")


   
    classifier = grid_search.best_estimator_
    return classifier



# def getNImportantFeatures(df, n = 20, attack_type = 'DOS', confusion : bool = False, graph: bool = False):
    
#     current_attack_type = f'attack_type_{attack_type}' 
#     df = moveColumnToEnd(df, current_attack_type)
#     df = df.rename(columns={current_attack_type : attack_type})
#     df = df.loc[:, ~df.columns.str.startswith('attack_type')]

#     df_X = df.iloc[:, :-1]
#     df_y = df.iloc[:, -1]

#     df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)



#     classifier = KNeighborsClassifier(n_neighbors = 5)
#     #classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#     classifier.fit(df_x_train, df_y_train)
#     df_y_pred = classifier.predict(df_x_test)

#     df_accuracy = accuracy_score(df_y_test, df_y_pred)


#     print(f'{attack_type} Dataset Accuracy: {df_accuracy * 100 :.2f}')
#     print(f'Recall: {recall_score(df_y_test, df_y_pred):.2f}')
#     print(f'F1 Score: {f1_score(df_y_test, df_y_pred):.2f}')

#     print("\nClassification Report:\n", classification_report(df_y_test,df_y_pred))

#     if confusion == True:
#         df_conf_matrix = confusion_matrix(df_y_test, df_y_pred)

#         plt.figure(figsize=(8, 6))
#         sns.heatmap(df_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Normal', f'{attack_type} Attack'], yticklabels=['Normal', f'{attack_type} Attack'])
#         plt.title(f'{attack_type} Attack Confusion Matrix')
#         plt.xlabel('Predicted Label')
#         plt.ylabel('True Label')
#         plt.show()

#     df_feature_importances = classifier.feature_importances_
#     df_feature_names = classifier.feature_names_in_

#     feature_importance_df = pd.DataFrame({'Feature': df_feature_names, 'Importance': df_feature_importances})
#     feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
#     feature_importance_df = feature_importance_df.head(n)

#     if graph == True:
        
#         plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'] )
#         plt.xlabel("Feature Importance")
#         plt.title(f"{attack_type} Feature Importance in Random Forest Classifier")
#         plt.show()



