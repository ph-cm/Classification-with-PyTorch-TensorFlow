#IRIS CLASSIFICATION

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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
