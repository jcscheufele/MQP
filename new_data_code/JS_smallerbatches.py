from sklearn.model_selection import train_test_split
import csv
import pandas as pd
from random import sample

data = pd.read_csv('/work/shared/DEVCOM-SC21/Network/data/trainingset_200m_greaterthan0.csv')

#print(data.describe())

data = data[data['alt_dif']>-200]
data = data[data['alt_dif']<0]

subset0= data.sample(n=25000)

subset0.to_csv("/work/shared/DEVCOM-SC21/Network/data/NYC_25000pix_200m_greaterthan0_new.csv")
