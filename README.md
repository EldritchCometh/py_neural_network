# Python Neural Network


## 👋 Description
This is a simple implementation of a neural network in Python. You can think of it as a translation of 3Blue1Brown's 4-part series on neural networks into computer code.

There are many libraries and projects out there for optimized machine learning. This project is about the human learning of machine learning. The code is intended to be well organized and as easy to follow and understand as possible. Every operation is broken up into discrete methods with descriptive names and inner workings that should be able to be understood if examined.

As a companion to the 3Blue1Brown series, I recommend watching that first or with this code open to see if you can follow along with his explanations in this code.
<br></br>
## 🚀 Use
### Default funtionality: 
If you don't change anything in the code, is to run like this:

```nn = NeuralNetwork([784, 16, 16, 10])```<br>
```nn.train_network(1, 32, 5)```

This will:
   * Create a NeuralNetwork object with shape `[784, 16, 16, 10]`
   * Train it `with learning rate 1`, `batch size 32`, and `duration 5 minutes`
   * Print reports evert 10 seconds and at the end of 5 minutes that include:
      - Accuracy of the predictions
      - Mean squared error measurement of loss

### Try out some of your own options:

   * Try different network shapes by changing the shape peramater of the NeuralNetwork class.
   * Try out different peramaters for `train_network()` for
      * learning rates
      * batch sizes, and 
      * training time
   * Save and load models by 
      * supplying a name to the neural_network class on instantiation to load and 
      * a name as an additional paramater of the train_network method to save.

I included a script for how I formatted and packaged the raw mnist data. I recommend just using the provided pickle but the script is there if you want to prepare the data yourself.
