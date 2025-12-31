from datetime import datetime
import pandas as pd 





path = r'C:\Users\Nyle\Documents\ResumeProjects\NetworkSecurityAI2\xzy.csv'
csv = pd.read_csv(path)
print(csv.head())





dt = datetime.now()
print(dt)