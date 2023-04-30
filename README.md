# Python Neural Network


## 👋 Description

This is a simple implementation of a neural network in Python. I intended it as a more or less rough translation of 3Blue1Brown's 4-part series on neural networks into code.

There are many libraries and projects out there for optimized machine learning. This project is about the human learning of machine learning. The code is intended to be well organized and as easy to follow as possible. Each operation is broken up into a discrete method with a descriptive name and inner workings that should be able to be understood if examined.

As a companion to the 3Blue1Brown series, I recommend watching that first or with this code open to see if you can follow along with his explanations in this code.

## 🚀 Use

### With no changes default functionality is to run like so:

```training_samples, testing_samples = IO.get_samples()```<br>
```structure = Architect.build([784, 64, 32, 10])```<br>
```network = Network(structure)```<br>
```evaluator = Evaluator(network, testing_samples)```<br>
```trainer = Trainer(network, training_samples, evaluator)```<br>
```trainer.train_network()```<br>
```IO.save_model(network, 'model01')```

**This will:**
   * Create a `NeuralNetwork` object with shape `[784, 16, 16, 10]`
   * Train it with `learning rate 0.01`, `batch size 1`, and `duration 1 minute`
   * Save the model as `model01` and
   * Print reports evert 10 seconds and at the end of the 1-minute duration containing:
      * 'Accuracy' of the predictions
      * 'Mean squared error' measurement of cost

### Try out some of your own options:
   * Try different network shapes by changing the shape parameter of the `NeuralNetwork` class.
   * Try out different parameters for `train_network()` with
      * 'learning rates'
      * 'batch sizes', and 
      * 'training times'
   * Save and load models by 
      * supplying a name to the neural_network class on instantiation to load and 
      * a name as an additional parameter of the train_network method to save.

## A word on `Architect`

The `Architect` class is probably the hardest part of the code to follow while, at the same time, being the least important to the aims and goals of the project. If you prefer, I think it is reasonable to treat it as a bit of it as a bit of a black box and take for granted the following explanation:
   * Its job is really just to construct an object-oriented model matching common images and descriptions of the structure of a neural network;
   * It makes a list of lists of `Neuron` objects,
   * It makes a list of lists of lists of `Connection` objects,
   * The **neuron** objects are iterated through and properly assigned references to each **connection** object,
   * The **connection** objects are iterated through and properly assigned references to each **neuron** object,
   * Neurons are assigned activation functions and
   * Weights are initialized to their starting values.

## 📝 Notes
   * Default values are tuned to work well for one minute of training. Different values would probably work better for achieving higher accuracy over a longer training duration.
   * Why is **"sigmoid squishification"** referenced repeatedly in the video but appears nowhere in this code?
      * I'm using **ReLU (Rectified Linear Unit)** activation here. You can find `relu_activation` it in `Architect`. It's simpler, more powerful and discussed in some detail at the end of video one.
   * What is `he_weight_init` in the `Architect` class? 
      * This is just a little function that takes the number of neurons in the previous layer and uses that to initialize an ideal starting weight for a connection in a ReLU activated network.
