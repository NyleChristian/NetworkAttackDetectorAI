Introduction:

/Any internal home network that has connection to the wider internet has a high level of risk to malicious outsider attacks. One solution is to keep home networks with valueble data completely isolated from the internet, though this is not satisfactory as files cannot be accessed remotely anymore. Instead of manually double-checking all suspicious activity, I thought to create an AI model that will notify me of suspicious behaviour on my network.

Problem Statement:

/Classify anomalous behavior on the network as either normal or one of 4 categories of attack types:

DoS, Probe, U2R, R2L

Solution:

/Use various AI models trained separately with differing feature importances and hyperparameters for each of the 4 attack types for maximum prediction accuracy per type

Process:

To begin with, I chose the NSL-KDD, a public dataset that is a nicely preprocessed form of the original 1999 KDD Cup dataset

The NSL-KDD stands as a standard benchmark used in the feasibility of network security solutions, AI or otherwise

Despite the NSL-KDD being a preprocessed form of another dataset, I still chose to preprocess the set in a couple ways to further improve my models accuracy:

