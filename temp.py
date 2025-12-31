# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder

# # Sample data
# data = {'City': ['New York', 'London', 'Tokyo', 'New York', 'Paris', 'London', 'New York']}
# df = pd.DataFrame(data)
# print(df)
# # Initialize the OneHotEncoder with default sparse output
# # In newer scikit-learn versions, use sparse_output=True for clarity, though it's the default
# encoder = OneHotEncoder(sparse_output=True)

# # Fit and transform the data
# encoded_sparse_data = encoder.fit_transform(df[['City']])
# print(encoder.categories_)
# # Print the sparse matrix representation
# print("Sparse Matrix Output:")
# print(encoded_sparse_data)

# # To view as a dense (regular NumPy) array if needed for specific operations
# # Be cautious with large datasets as this can consume a lot of memory
# encoded_dense_data = encoded_sparse_data.todense()
# print("\nDense Array Output (use with caution on large data):")
# print(encoded_dense_data)

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Sample DataFrame
df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'], 
                   'Price': [10, 20, 15, 12, 18]})

# Initialize the encoder and fit/transform the 'Color' column
# Reshape the column to a 2D array as scikit-learn expects
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_data = encoder.fit_transform(df[['Color']])

# Get the feature names for the new columns
# In newer scikit-learn versions, this is get_feature_names_out()
encoded_cols = encoder.get_feature_names_out(['Color'])

# Create a DataFrame from the encoded data
df_encoded = pd.DataFrame(encoded_data, columns=encoded_cols)

# Concatenate the new DataFrame with the original one (excluding the original 'Color' column)
df_final = pd.concat([df.drop('Color', axis=1), df_encoded], axis=1)

print(df_final)

