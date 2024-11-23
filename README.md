# Classification-with-PyTorch-TensorFlow

# Part I: Iris Classification

The Iris dataset contains 150 records of 3 different classes of irises. Each record contains 4 numeric parameters: sepal length/width and petal length/width. It is an example of a simple dataset, for which you do not need a powerful neural network.

---

## Getting the Dataset

The Iris dataset is built into Scikit Learn

## Visualize the Data

In many cases, it makes sense to visualize the data to see if they look separable. This assures us that we should be able to build a good classification model. Because we have a few features, we can build a series of pairwise 2D scatter plots, showing different classes by different dot colors. This can be automatically done by a package called **seaborn**

## Normalize and Encode the Data

To prepare data for neural network training, we need to normalize inputs in the range [0..1]. This can be done either using plain `numpy` operations or **Scikit Learn** methods.

Also, you need to decide if you want the target label to be one-hot encoded or not. **PyTorch** and **TensorFlow** allow you to feed in class numbers either as an integer (from 0 to N-1) or as a one-hot encoded vector. When creating a neural network structure, you need to specify the loss function accordingly (e.g., `sparse categorical crossentropy` for numeric representation, and `crossentropy loss` for one-hot encoding). One-hot encoding can also be done using Scikit Learn


---

## Split the Data into Train and Test

Since we do not have separate train and test datasets, we need to split it into train and test datasets using **Scikit Learn**.

---

# Part 2: MNIST Training

Both **Keras** and **PyTorch** contain MNIST as a built-in dataset, so you can easily get it with a couple of lines of code ([Keras](https://keras.io/), [PyTorch](https://pytorch.org/)). You will also be able to load both train and test datasets without manually splitting them.
