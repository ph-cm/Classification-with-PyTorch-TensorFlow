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