#IRIS CLASSIFICATION

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

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

#Define and Train Neural Network

#Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

#Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size = 16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size= 16, shuffle=False)

#Define the Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork,self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
#Initialize the network
input_size = X_train.shape[1] #4features
hidden_size = 10              #NUmber of neurons in the hidden layer
output_size = 3               #3classes in Iris dataset
model = NeuralNetwork(input_size, hidden_size, output_size)

#Definning loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#training loop
num_epochs = 50
train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_losses.append(running_loss / len(train_loader))
    
    #validation accuracy
    model.eval()
    y_val_pred = []
    y_val_true = []
    with torch.no_grad():
        for x_val, y_val in test_loader:
            outputs = model(x_val)
            _, predicted = torch.max(outputs, 1)
            y_val_pred.extend(predicted.numpy())
            y_val_true.extend(y_val.numpy())
    val_acc = accuracy_score(y_val_true, y_val_pred)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    
#Plotting Training loss and Validation Accuracy
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epoch")
plt.legend()

plt.subplot(1,2,2)
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy Over Epoch")
plt.legend()

plt.show()

#test the model
model.eval()
y_test_pred = []
with torch.no_grad():
    for X_test_batch, _ in test_loader:
        outputs = model(X_test_batch)
        _, predicted = torch.max(outputs, 1)
        y_test_pred.extend(predicted.numpy())
test_accuracy = accuracy_score(Y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")