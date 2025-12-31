import pandas as pd
import sklearn as sk


path = r'C:\Users\Nyle\Documents\ResumeProjects\NetworkSecurityAI2\kdd_test.csv'

df = pd.read_csv(path)

print(df['labels'].value_counts())

