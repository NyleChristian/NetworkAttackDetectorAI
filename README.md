# Introduction:

# Problem Statement:

Classify anomalous behavior on the network as either normal network traffic or one of 4 categories of attack types:

DoS, Probe, U2R, R2L

# Solution:

Use various AI models trained separately with differing feature importances and hyperparameters for each of the 4 attack types for maximum prediction accuracy per type

## Process:

1. **Data Set**

   * NSL-KDD public dataset was chosen for model training
   * The dataset is considered as an industry benchmark used in the feasibility of network security AI solutions
2. **Preprocessing** - *Preparing and restructuring data to allow the model to most efficiently learn from it*

   * Dropped duplicate rows and null entries
   * Simplified 36 attack labels into 4 generalized attack types or as normal network activity
   * **One-Hot Encoded** attack types and all other categorical feature columns. While one-hot encoding requires more columns than label encoding, it simplifies the training/prediction process as there are simple binary columns indicating normal or attack, especially useful for models such as Random Forest, which are tree-based algorithms
   * Drop categorical columns after one-hot encoding. This is because models perform better with purely numerical data and they are now redundant after encoding
   * Split the dataset into four groups based on attack type, with each group containing normal entries for the model to compare against. This allows for increased model accuracy as we will use Random Forest to determine four sets of the most important features for classifying each type
3. **Feature Extraction Based on Importance** - *Find distinct influential features for each attack type*

   * Used sklearn's Random Forest Classifier on each of the four datasets to get a list of the top 20 most influential features
   * Each attack type is given its own list of most influential features, which helps model accuracy and efficiency as the model only needs to consider 20 features instead of the ~40 total features, and is more specialized than a general model that might focus on the same set of important features for all four attack types
     <img src="assets/20260104_162219_FeatureImportancev1.png" width="75%" height="75%" />
4. Hyperparameter Tuning

   * Used GridSearchCV to brute force determine the best values of Random Forest hyperparameters
   * While GridSeachCV is computationally expensive, but it provides better results compared to RandomSearchCV
   * Surprisingly, the addition of hyperparameter tuning via GridSearchCV provided no significant improvement in accuracy worth mentioning
5. Training Model

   * Models used: Random Forest, KNN, and Gaussian NB
   * Best results were found with random forest algorithm
   * Final results:


   | attack_type | precision | recall | f1-score | support |
   | ------------- | ----------- | -------- | ---------- | --------- |
   | DOS         | 0.98      | 0.94   | 0.96     | 1625    |
   | PROBE       | 0.89      | 0.85   | 0.87     | 453     |
   | R2L         | 0.99      | 0.91   | 0.95     | 607     |
   | U2R         | 1.00      | 0.79   | 0.88     | 19      |

# Challenges

1. Low volume of test data for certain attack types
   * R2L and U2R both had very low occurrences in their own datasets. This led the model to struggle to find strong feature importances and other correlations in data, hurting model accuracy significantly as it would consider all entries to be normal whether they actually were or not <img src="assets/beforeafter" width="75%" height="75%" />
   * To solve this, a custom training/test dataset split was used that specifically ensured a significant amount of instances of an attack type are present within its own dataset.
   * As seen in the image, being conscious of data distributions led to a significant overall increase in f1 scores of attack identification for both U2R and R2L
   *
