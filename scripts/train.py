import pandas as pd
import glob as gb

#Load dataset
data=r"D:\RealTime_Alert_Analysis\dataset"

#club all files into one
files=gb.glob(f"{data}/*.pcap_ISCX.csv")
df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)

# print(df.columns.tolist())


#Encoding labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df[' Label'] = le.fit_transform(df[' Label'])

# print("Successfully! Labels encoded.")
# print(dict(zip(le.classes_, le.transform(le.classes_))))


#Split dataset into features and labels

X = df.drop(' Label', axis=1)
y = df[' Label']

# print(X,y)
# print(X.shape, y.shape)

import numpy as np

# Replace inf and -inf with NaN
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna()
y = y.loc[X.index]

#Splitting dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# print("Training samples:", X_train.shape[0])
# print("Testing samples:", x_test.shape[0])

#Training the model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("✅ Random Forest model trained successfully!")

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))