# EmotionsRealTimeML
Using Python Machine Learning and external camera program upon the successfull training recognizes emotions on human faces in real time.

## Project Goal
The aim of the project is to create a program that allows to detect the current emotion on the face of a person visible on the camera and through an uploaded photograph. The program was created based on the CNN (Convolutional Neural Networks) model, which classify individual features from images. The following libraries were used in the project:
- Matplotlib.pyplot
- OpenCV
- Os
- Tensorflow (keras):
  - Keras.callbacks
  - Keras.models
  - Keras.optimizer_v2.adam
  - Keras.preprocessing_image
  - Keras.layers
- DeepFace

In order to teach the model, a database of images from the following website was used: [Kaggle-Face-Expression-Recognition-Dataset](https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset). This database consists of 35,900 images divided into validation and training subfolders. In the training folder the photos were divided into individual emotions:
- Angry - 3993 photos
- Disgust - 436 photos
- Fear - 4103 photos
- Happy - 7164 photos
- Neutral - 4982 photos
- Sad - 4938

**The repository also contains the model.h5 file, which is essential for training models.**

## Project Structure
The project is divided into two sub projects. One of them contains a script to train the data and create a model, while the other project reads the model and based on it performs the main idea of the project, which is to detect emotions on a person's face.

## Data training
To train the data, the image size was set to 48px by 48px and the batch size was set to 128. The following layers were used for the model: 
- Convolutional Layer
- Pooling Layer
- Batch Normalization
- MaxPool2D
- Dropout
- Flatten
- Fully Connected Layer

The model was then compiled with the defined optimizer "Adam" with learning_rate set to 0.0001, loss parameter set to "categorical_crossentropy and metrics set to "accuracy".

## Fitting the model to training and validation data
For model validation, a checkpoint variable was set to call back the ModelCheckpoint method, used in conjunction with learning using model.fit_generator() to save the model or weights (in a checkpoint file, e.g. model.h5) at some interval, so that the model or weights can be loaded later to continue learning from the saved state. 48 epochs are also defined so that the program can get the best possible precision result.

## Determination of precision and lost data
The program was written to train 48 models and if there is no significant increase in the value of "val_accuracy" the training process is to stop.
