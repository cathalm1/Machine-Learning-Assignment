import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from mlxtend.plotting import plot_confusion_matrix

# Load training set
dataset = np.loadtxt('train-io.txt')
# Training 70% of test-io
X = dataset[0:210000,:-1]
Y = dataset[0:210000:,-1]
# Testing 30% test-io
A = dataset[210000:,:-1]
B = dataset[210000:,-1]

classifier = Sequential()
# First Hidden Layer
classifier.add(Dense(140, activation='swish', input_dim = len(X[0,:])))
# Second  Hidden Layer
classifier.add(Dense(120, activation='swish'))
# Third  Hidden Layer
classifier.add(Dense(100, activation='swish'))
# Fourth  Hidden Layer
classifier.add(Dense(60, activation='swish'))
# Fifth  Hidden Layer
classifier.add(Dense(40, activation='swish'))
# Sixth  Hidden Layer
classifier.add(Dense(20, activation='swish'))
# Output Layer
classifier.add(Dense(1, activation='sigmoid'))
# Classifier summary
classifier.summary()

def train():
    # Compile the network
    classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
    # Fit the data to the training dataset
    classifier.fit(x = X, y = Y, verbose = 1, epochs=100)
    # Save network weights
    classifier.save_weights('weights.h5', save_format='h5')

def predict_test_in_text():
    # Load weights
    classifier.load_weights('weights.h5')
    # Load test set
    testset = np.loadtxt('test-in.txt')
    # Prediction
    prediction = classifier.predict(testset)
    # Threshold value calculated during 30% test set (commented code below)
    prediction[prediction > 0.4723097] = int(1)
    prediction[prediction <=0.4723097] = int(0)
    # Save prediction
    np.savetxt('test-out.txt', prediction, fmt="%i")
    # View results
    testout = np.loadtxt('test-out.txt')
    print("0's: ", np.sum(1-testout))
    print("1's: ", np.sum(testout))


def predict_30_percent_testing_set():
    # Load weights
    classifier.load_weights('weights.h5')
    # Prediction (Variables A and B are 30% IO of train-io.txt)
    prediction = classifier.predict(A)
    # Calculate Accuracy
    accuracy = accuracy_score(B, prediction.round())
    # Calculate Precision
    precision = precision_score(B, prediction.round())

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("0's: ", np.sum(1-prediction))
    print("1's: ", np.sum(prediction))
    print("B 0's", np.sum(1-B))
    print("B 1's", np.sum(B))

    # Construct confusion matrix
    cf = confusion_matrix(B, prediction.round())
    fig, ax = plot_confusion_matrix(conf_mat = cf, figsize=(6,6), show_normed = False)
    plt.tight_layout()
    fig.savefig("confusion_matrix.png")
    # Construct ROC curve
    fpr, tpr, threshold = roc_curve(B, prediction)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
    lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    # Calculate Threshold
    optimal_index = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_index]
    print("Threshold: ", optimal_threshold)

if __name__ == "__main__":
    #train()
    #predict_30_percent_testing_set()
    predict_test_in_text()
