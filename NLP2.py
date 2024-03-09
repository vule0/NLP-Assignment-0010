import os
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import csv
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
def preprocess_review(sentence):
   # sentence_array = sentence.split()
   t_lower = sentence.lower()

   # Removing HTML tags
   t_clean = re.sub(r'<.*?>', '', t_lower)

   # Removing Noise
   t_clean = re.sub(r'[^a-zA-Z\s]', '', t_clean)

   # Stopwords Removal
   word_tokens = word_tokenize(t_clean)
   t_stop = [word for word in word_tokens if word not in stop_words]

   # Stemming (using Porter Stemmer)
   t_stem = [stemmer.stem(word) for word in t_stop]

   return ' '.join(t_stem)


review_data = []
with open("Assignment_2_modified_ Dataset.csv", 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    
    header = next(csv_reader)
    
    for review, sentiment in csv_reader:
      processed_review = preprocess_review(review)
      # print(processed_review)
      review_data.append([processed_review, sentiment])
      # review_data.append(row)


# processed_data = []
# for review, sentiment in review_data:
#    processed_review = preprocess_review(review)
#    processed_data.append([processed_review, sentiment])



print(review_data[1])
# print(processed_data[1])
np.random.seed(1234)
tf.random.set_seed(1234)

#### Load your Train, Validation and Test data_size set
data_size, train_ratio, val_ratio, test_ratio = len(review_data), .80, .10, .10
train_size = int(train_ratio * data_size)
val_size = int(val_ratio * data_size)
test_size = data_size - train_size - val_size
np.random.shuffle(review_data)

train_data = review_data[:train_size]
val_data = review_data[train_size:train_size + val_size]
test_data = review_data[train_size + val_size:]

# Split into features (X) and labels (y)
X_train, y_train = zip(*train_data)
X_val, y_val = zip(*val_data)
X_test, y_test = zip(*test_data)

all_words = [word for review in X_train for word in review]

label_encoder = LabelEncoder()
label_encoder.fit(all_words)
X_train = [label_encoder.transform(review) for review in review_data]

X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)

## Normalize your splits
size_input = X_train.shape[0]
# print(size_input)
size_hidden1 = 128
size_hidden2 = 128
size_hidden3 = 128
# size_output = 10
size_output = y_train.shape[0]

number_of_train_examples = X_train.shape[0]
number_of_test_examples = X_test.shape[0]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)


num_classes = len(np.unique(y_train_encoded))
print(num_classes)
y_train = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)
y_val = tf.keras.utils.to_categorical(y_val_encoded, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)



# Define class to build mlp model
## Change this class to add more layers
class MLP(object):
 def __init__(self, size_input, size_hidden1, size_hidden2, size_hidden3, size_output, embedding_dim, device=None):
    """
    size_input: int, size of input layer
    size_hidden1: int, size of the 1st hidden layer
    size_hidden2: int, size of the 2nd hidden layer
    size_output: int, size of output layer
    device: str or None, either 'cpu' or 'gpu' or None. If None, the device to be used will be decided automatically during Eager Execution
    """
    self.size_input, self.size_hidden1, self.size_hidden2, self.size_hidden3, self.size_output, self.device =\
    size_input, size_hidden1, size_hidden2, size_hidden3, size_output, device

    self.embedding = Embedding(input_dim=size_input, output_dim=embedding_dim, input_length=None)
    # Initialize weights between input mapping and a layer g(f(x)) = layer
    self.W1 = tf.Variable(tf.random.normal([embedding_dim, self.size_hidden1],stddev=0.1)) # Xavier(Fan-in fan-out) and Orthogonal
    # Initialize biases for hidden layer
    self.b1 = tf.Variable(tf.zeros([1, self.size_hidden1])) # 0 or constant(0.01)

    # Initialize weights between input layer and 1st hidden layer
    self.W2 = tf.Variable(tf.random.normal([self.size_hidden1, self.size_hidden2],stddev=0.1))
    # Initialize biases for hidden layer
    self.b2 = tf.Variable(tf.zeros([1, self.size_hidden2]))

     # Initialize weights between 1st hidden layer and output layer
    self.W3 = tf.Variable(tf.random.normal([self.size_hidden2, self.size_output],stddev=0.1))
    # Initialize biases for output layer
    self.b3 = tf.Variable(tf.zeros([1, self.size_output]))

    # Define variables to be updated during backpropagation
    self.variables = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]

 def forward(self, X):
    """
    forward pass
    X: Tensor, inputs
    """
    if self.device is not None:
      with tf.device('gpu:0' if self.device=='gpu' else 'cpu'):
        self.y = self.compute_output(X)
    else:
      self.y = self.compute_output(X)

    return self.y

 def loss(self, y_pred, y_true):
    '''
    y_pred - Tensor of shape (batch_size, size_output)
    y_true - Tensor of shape (batch_size, size_output)
    '''
    #y_true_tf = tf.cast(tf.reshape(y_true, (-1, self.size_output)), dtype=tf.float32)
    y_true_tf = tf.cast(y_true, dtype=tf.float32)
    y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss_x = cce(y_true_tf, y_pred_tf)
    # Use keras or tf_softmax, both should work for any given model
    # loss_x = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred_tf, labels=y_true_tf))

    return loss_x

 def backward(self, X_train, y_train):
    """
    backward pass
    """


    with tf.GradientTape() as tape:

      predicted = self.forward(X_train)
      current_loss = self.loss(predicted, y_train)


    grads = tape.gradient(current_loss, self.variables)

    return grads


 def compute_output(self, X):
    """
    Custom method to obtain output tensor during forward pass
    """
    # Cast X to float32
    X_tf = tf.cast(X, dtype=tf.float32)
    # X_tf = X

    # Compute values in hidden layers
    h1 = tf.matmul(X_tf, self.W1) + self.b1
    z1 = tf.nn.relu(h1)

    h2 = tf.matmul(z1, self.W2) + self.b2
    z2 = tf.nn.relu(h2)


    # Compute output
    output = tf.matmul(z2, self.W3) + self.b3

    #Now consider two things , First look at inbuild loss functions if they work with softmax or not and then change this
    # Second add tf.Softmax(output) and then return this variable
    return (output)

# Set number of epochs
NUM_EPOCHS = 100
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
embedding_dim = 2
# Initialize model using CPU

mlp_on_cpu = MLP(size_input, size_hidden1, size_hidden2, size_hidden3, num_classes, embedding_dim, device='cpu')

time_start = time.time()

for epoch in range(NUM_EPOCHS):
    
    y_pred = mlp_on_cpu.forward(X_train)

    y_pred_softmax = tf.nn.softmax(y_pred)

    # Calculate the loss
    loss = mlp_on_cpu.loss(y_pred_softmax, y_train)

    # Backpropagation
    grads = mlp_on_cpu.backward(X_train, y_train)

    # Update weights
    optimizer.apply_gradients(zip(grads, mlp_on_cpu.variables))
    # # Calculate the loss
    # loss = mlp_on_cpu.loss(y_pred, y_train)

    # # Backpropagation
    # grads = mlp_on_cpu.backward(X_train, y_train)

    # # Update weights
    # optimizer.apply_gradients(zip(grads, mlp_on_cpu.variables))

    # Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss:.3f}")
    # pass



time_taken = time.time() - time_start

