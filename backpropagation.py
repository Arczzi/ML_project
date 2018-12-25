import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')
#data.info()
data.columns
X = data[['sex', 'address', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'traveltime',
       'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities',
       'nursery', 'higher', 'internet', 'freetime', 'goout', 'health',
       'absences']]
Y = data[['G1','G2','G3']]

X.info()
Y.info()
