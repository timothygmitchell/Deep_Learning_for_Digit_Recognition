# Deep_Learning_for_Digit_Recognition

This project showcases a deep learning solution to the 'Digit Recognizer' Kaggle competition using the famous MNIST data set. The challenge is a classic computer vision problem with the goal of classifying handwritten digits.

I used [Poonam Ligade's Kaggle notebook](https://www.kaggle.com/poonaml/deep-neural-network-keras-way) as a starting point. 

Image pre-processing involved centering and scaling pixels to unit variance. For activation functions, optimizers, and other technical details, I enriched my code with many comments.

In order of increasing complexity, I tested 5 different architectures:
- a simple fully connected feedforward neural network (single-layer perceptron)
  - 0.9079 validation accuracy after 3 epochs
- a fully connected feedforward neural network with one hidden layer
  - 0.9600 validation accuracy after 1 epoch
- a deep convolutional neural network
  - 0.9714 validation accuracy after 1 epoch
- a deep convolutional neural network with data augmentation
  - 0.9629 validation accuracy after 1 epoch
- a deep convolutional neural network with batch normalization
  - 0.9993 train accuracy after 3 epochs
  - 0.9927 accuracy on the holdout set

Data augmentation did not improve performance, but batch normalization achieved 99.27% accuracy, which is in the top 25% of all submissions (793/3166). 

Increasing the number of epochs or joining together mutiple CNNs might improve accuracy further.
