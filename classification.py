#IRIS CLASSIFICATION

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#Getting the dataset
iris = load_iris()
features = iris['data']
labels = iris['target']
class_names = iris['target_names']
feature_names = iris['feature_names']

print(f'Features: {feature_names}, Classes: {class_names}')

#Visualize the data
df = pd.DataFrame(features, columns=feature_names).join(pd.DataFrame(labels, columns=['Label']))
print(df)

sns.pairplot(df,hue='Label')
plt.show()

#Normalize and Encode the Data
# Normalize the features using L1 norm
features_normalized_l1 = normalize(features, norm="l1", axis=1)

# Normalize the features using L2 norm
features_normalized_l2 = normalize(features, norm="l2", axis=1)

# Convert to DataFrame for visualization
df_l1 = pd.DataFrame(features_normalized_l1, columns=feature_names).assign(Label=labels)
df_l2 = pd.DataFrame(features_normalized_l2, columns=feature_names).assign(Label=labels)

# Visualize L1 normalized data
print("L1 Normalized DataFrame:")
print(df_l1.head())

# Visualize L2 normalized data
print("L2 Normalized DataFrame:")
print(df_l2.head())

# Pairplot for L1 normalized data
sns.pairplot(df_l1, hue='Label')
plt.show()

# Pairplot for L2 normalized data
sns.pairplot(df_l2, hue='Label')
plt.show()

#Split the Data into Train and Test
#Split the data into train and test sets(80%train, 20%test)
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

#Normalize the train and test sets using L2 normalization
X_train_normalized = normalize(X_train, norm="l2", axis=1)
X_test_normalized = normalize(X_test, norm="l2", axis=1)

#Convert to DataFrames for visualization
df_train = pd.DataFrame(X_train_normalized, columns=feature_names).assign(Label=Y_train)
df_test = pd.DataFrame(X_test_normalized, columns=feature_names).assign(Label=Y_test)

#Visualize the trainning data
sns.pairplot(df_train, hue='Label')
plt.title('L2 Normalized Training Data')
plt.show()

#Visualize the testing data
sns.pairplot(df_test, hue='Label')
plt.title('L2 Normalized Test Data')
plt.show()

