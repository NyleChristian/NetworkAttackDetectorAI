Introduction:

/Any internal home network that has connection to the wider internet has a high level of risk to malicious outsider attacks. One solution is to keep home networks with valueble data completely isolated from the internet, though this is not satisfactory as files cannot be accessed remotely anymore. Instead of manually double-checking all suspicious activity, I thought to create an AI model that will notify me of suspicious behaviour on my network.

Problem Statement:

/Classify anomalous behavior on the network as either normal or one of 4 categories of attack types:

DoS, Probe, U2R, R2L

Solution:

/Use various AI models trained separately with differing feature importances and hyperparameters for each of the 4 attack types for maximum prediction accuracy per type

Process:

1. Data Set
   * I chose the NSL-KDD public dataset,
   * The dataset is considered as an industry benchmark used in the feasibility of network security AI solutions
2. Preprocessing
   * I dropped duplicate rows and null entries
   * Simplified 36 attack labels into 4 generalized attack types or normal
   * One-Hot Encoded attack types and all other categorical feature columns. While one-hot encoding requires more columns than label encoding, it simplifies the training/prediction process as there are simple binary columns indicating normal or attack, especially useful for models such as Random Forest, which are tree-based algorithms
   * Drop categorical columns after one-hot encoding. This is because models perform better with purely numerical data
   * Split the dataset into four groups based on attack type, with each group containing normal entries for the model to compare against. This allows for increased model accuracy as we will use Random Forest to determine four sets of the most important features for classifying each type
   *
3. Feature Extraction Based on Importance
   * I used sklearn's Random Forest Classifier on each of the four datasets to get a list of the top 20 most influential features used to determine attack type
   * Each attack type is given its own list of most influential features, which helps model accuracy and efficiency as the model only needs to consider 20 features instead of the ~40 total features, and is more specialized than a general model that might focus on the same set of important features for all four attack types
   *  <img src="assets/20260104_162219_FeatureImportancev1.png" width="50%" height="50%" />


4. Hyperparameter Tuning
5.
6. Training Model
7.

* To begin with, I chose the NSL-KDD, a public dataset that is a nicely preprocessed form of the original 1999 KDD Cup dataset
*
* Despite the NSL-KDD being a preprocessed form of another dataset, I still chose to preprocess the set in a couple ways to further improve my models accuracy:

1. I started with dropping duplicate rows and rows with null entries, as these can negatively impact model accuracy
