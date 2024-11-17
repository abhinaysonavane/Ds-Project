#Step 1: Import Libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns



# Step 2: Load Dataset from CSV
df = pd.read_csv('Iris.csv')

# Check the first few rows of the dataset
print(df.head())

df = df.drop(columns = ['Id'])
print(df.head())
#to show stat about data
print(df.describe())
#info about data type
print(df.info())
#no.of sample on each class
print(df['Species'].value_counts())
#to check null values in dataset
print(df.isnull().sum())

# Step 3: Visualize Data
# 1. scatter plot
"""colors = ['red','orange','blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']
for i in range(3):
    x =df[df['Species']==species[i]]
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'],c = colors[i], label=species[i])
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend()
    plt.show()
    plt.savefig('plot.png')"""

# 2. Histograms for each feature (excluding 'species')
"""(df['SepalLengthCm'].hist())
plt.legend()
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Histogram of SepalLength ")
plt.show()
plt.savefig('plot.png')"""
# 3.box plot 
"""df.boxplot(by='SepalLengthCm', column=['SepalWidthCm'], grid=False, fontsize=8)
plt.title("Box Plot of SepalWidth by SepalLength ")
plt.xlabel("SepalLength")
plt.ylabel("SepalWidth")
plt.show()
plt.savefig('plot.png')"""



# Calculate the correlation matrix (dropping non-numeric columns, e.g., 'Species')
"""corr_matrix = df.drop(columns='Species').corr()

# Plot the heatmap using seaborn
plt.figure(figsize=(8, 6))  # Set the figure size
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Iris Features')

# Save the plot to a file
plt.savefig('correlation_matrix.png')
print("Plot saved as 'correlation_matrix.png'")

# Show the plot
plt.show()"""
#label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
print(df.head())
#model training
from sklearn.model_selection import train_test_split
#train -70
#test-30
x =df.drop(columns=['Species'])
y = df['Species']
X_train, X_test,y_trian,y_test = train_test_split(x,y,test_size=0.30)
#logistic regrassion
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
#model training
model.fit(X_train,y_trian)
#print metric to get performance
print("Accuracy:",model.score(X_test,y_test)*100)