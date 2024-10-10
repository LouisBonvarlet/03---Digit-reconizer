import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import itertools

import tensorflow as tf
from tensorflow.keras.layers import Input  # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import SGD,Adam # type: ignore
from keras.optimizers import RMSprop # type: ignore
from keras.callbacks import ReduceLROnPlateau # type: ignore

# Train-validation splitting
X_train2,X_validation,y_train2,y_validation = train_test_split(X_train,y_train, # type: ignore
                                                             test_size=0.3,random_state=0)

# Cnvertion of the input images into the proper format for convolutional neural network 
# Convolution expects height x width x color
X_train2 = np.expand_dims(X_train2,-1)
X_validation = np.expand_dims(X_validation,-1)
print(X_train2.shape, X_validation.shape)

# Define the optimizer
optimizer = RMSprop(learning_rate=0.001,rho=0.9,epsilon=1e-08)

# Defining the learning rate reduction routine
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0000001)

# Defining the checkpoint callback
checkpoint_path = "results.weights.h5"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                      save_best_only=True,
                                      monitor='val_accuracy',
                                      mode='max',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_freq='epoch')

##MODEL##
# Set random seed
tf.random.set_seed(42)

DROPOUT = 0.4

model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=X_train[0].shape),   # type: ignore
        tf.keras.layers.RandomRotation(0.025),
        tf.keras.layers.RandomZoom(0.025),
        tf.keras.layers.RandomTranslation(height_factor=0.125, width_factor=0.125), 
        tf.keras.layers.RandomContrast(factor=0.125),
        ### First block of layers ###
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(DROPOUT),
        ### Second block of layers ###
        tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(DROPOUT),
        ### Third block of layers ###
        tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(DROPOUT),
        ### Fourth block of layers ###
        tf.keras.layers.Conv2D(256, (3,3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3,3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(DROPOUT),
        ### Final layers before output ###
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(10, activation="softmax")
])

##MODEL FITTING##
# Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer=optimizer,            
                 metrics=["accuracy"])

# Fit the model (to the normalized data)
history = model.fit(X_train2,y_train2,
                      epochs=100,
                      batch_size=32,
                      validation_data=(X_validation,y_validation), 
                      callbacks=[learning_rate_reduction,checkpoint_callback])


model.summary()

## Result Analyse ##
#Loss Curve 
plt.plot(history.history['loss'],label='train loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.xlabel('epochs',fontsize=12)
plt.ylabel('loss',fontsize=12)
plt.legend()

plt.title('Train and Validation Losses',fontsize=20)

plt.tight_layout()
plt.gca().set_facecolor('gainsboro')
plt.gcf().patch.set_facecolor('lightsteelblue')

#Acurracy curves 
plt.plot(history.history['accuracy'],label='train accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('epochs',fontsize=12)
plt.ylabel('accuracy',fontsize=12)
plt.legend()

plt.title('Train and Validation Accuracies',fontsize=20)

plt.tight_layout()
plt.gca().set_facecolor('gainsboro')
plt.gcf().patch.set_facecolor('lightsteelblue')

#Results on validation data 
score = model.evaluate(X_validation,y_validation,verbose=0)
print('Test Loss: {:.4f}'.format(score[0]))
print('Test Accuracy: {:.4f}'.format(score[1]))

#Model parameter with the best accuracy 
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(X_validation,y_validation,verbose=0)

print('Restored model')
print('Test Loss: {:.4f}'.format(loss))
print('Test Accuracy: {:.4f}'.format(acc))

#Confusion Matrix 
def make_confusion_matrix(y_true,y_pred,classes=None,figsize=(10,10),text_size=15):
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.

       If classes is passed, confusion matrix will be labelled, if not, integer class values
       will be used.

       Args:
         y_true: Array of truth labels (must be same shape as y_pred).
         y_pred: Array of predicted labels (must be same shape as y_true).
         classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
         figsize: Size of output figure (default=(10,10)).
         text_size: Size of output figure text (default=15).

       Returns:
         A labelled confusion matrix plot comparing y_true and y_pred.

       Example usage:
         make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15,15),
                          text_size=10)
    """
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:,np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes),
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)

  # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

  # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i,j]} ({cm_norm[i,j]*100:.1f}%)",
             horizontalalignment="center",
             color="white" if cm[i,j] > threshold else "black",
             size=text_size)
        
# These are the predicted class probabilities (probs)
y_probs = model.predict(X_validation)

# I am printing the first prediction.
y_probs[0]

# The probs need to be converted in the predicted labels 
# by choosing the highest probability in the list
pred_labels = y_probs.argmax(axis=1)


#Confusion Matrix Plot 
classes = list(y_train.unique())  # type: ignore
classes.sort()
classes = [str(x) for x in classes]

make_confusion_matrix(y_true=y_validation,
                      y_pred=pred_labels,
                      classes=classes,
                      figsize=(15,15),
                      text_size=10)

#Identification of the misclassified items 
misclassified_idx = np.where(y_validation != pred_labels)[0]

plt.figure(figsize=(10,10))

for i in range(9):
    idx = np.random.choice(misclassified_idx)
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_validation[idx].reshape((28,28)),cmap='gray')
    label_index = int(pred_labels[idx])
    plt.title(f'Pred label: {classes[pred_labels[idx]]}',fontsize=8)
    #plt.title(f'True label: {classes[y_validation[idx]]}; Pred label: {classes[pred_labels[idx]]}',fontsize=8)
plt.show()


## PREDICTION ##
X_test = np.expand_dims(X_test,-1) # type: ignore

# predict results
results = model.predict(X_test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("../03-Submission/LouisBonvarlet_CNN_DigitRecogniser.csv",index=False)