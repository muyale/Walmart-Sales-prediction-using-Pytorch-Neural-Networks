# Walmart-Sales-prediction-using-Pytorch-Neural-Networks

![image](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/8c35f5e0-b6b9-4381-8bbe-68899d2e8adc)

Walmart Inc.  formerly Wal-Mart Stores, Inc. is an American multinational retail corporation that operates a chain of hypermarkets (also called supercenters), discount department stores, and grocery stores in the United States, headquartered in Bentonville, Arkansas. Given its sales data our task is to analyse the data then using machine learning create a model that predicts future sales. For this project I will use Deep Learning to create models to do this for us . What then is deep learning?

Deep learning is a method in artificial intelligence (AI) that teaches computers to process data in a way that is inspired by the human brain. Deep learning models can recognize complex patterns in pictures, text, sounds, and other data to produce accurate insights and predictions.The most fascinating thing about deep learning is how the model learns.,see in 
deep learning the output of one network becomes the input of the other network. Deep learning is based on what is termed as neural networks which have several layers ,namely the input layer
the hidden layers and output layers . Here's the illustration

![image](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/af270af8-c021-436d-a7b9-bf7a96843300)


There are several neural networks such as ANNs,CNNS ,LSTMS and GRUs with each being used in a  specific problem. For this project my main focus is RNNS ,LSTMS and GRUS.
First I imported some dependencies 

![Importing dependecies](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/439a9e8f-53f1-4d01-acf6-8bc4d51aa8a1)

Basic statistics tell us a little about our data ,they help up build some intution on our data

![Basic statistics](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/8041766c-101d-4d57-8a9f-a9f462d57985)

I then proceeded to create a plotter class that I would use to generate plots ,and avoid repetition of code:

![Plotter Class oop](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/dc159baf-d2b3-4ef7-87f2-e9c0b77ab89f)

For univariate data histograms easily capture the underlying relationships ,for example the following 

![Histogram 1](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/1e20ab00-6f85-4d93-bc08-2fb5cc9c1b14)

![Histogram 2](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/6290ce46-6a19-4d8b-a952-113d1f4278d0)


Lineplots can easily show the trend of variables over time ,I looked at the Sales trend 

![Sales line plot](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/5e55d971-2032-428c-a325-2e6379f51641)

Fuel prices have an effect on the economy ,if the fuel prices are high the prices will be high and this means that the sales will plummet

![Fuel prices](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/39fcb474-a57c-4ea9-9570-bc7c5f260718)

The next step was machine learning and so I had to select the feature and target variables 

![Selecting X and Y](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/8ca0e7da-cd64-4f62-afe7-f1748690bc4c)

As a rule of thumb we are required to split the data into training and testing data ,also Scaling prevents overfitting before fitting a machine learning model I did these steps

![Train test split](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/3bcbaa56-115d-453b-a764-e8f5b9ea8391)

My first machine learning was Linear Regression ,this type of machine learning algorithm tries to find linear relationship bewteen variables.For validation I used Mean Squared Error

![Linear Regression](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/1f4cdaa8-13af-4e63-b788-a4384b0a0ccd)

Decision Tree is a decision-making tool that uses a flowchart-like tree structure or is a model of decisions and all of their possible results, including outcomes, input costs, and utility.Decision-tree algorithm falls under the category of supervised learning algorithms. It works for both continuous as well as categorical output variables.

![Decision Tree Regressor](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/cc097dd8-6b05-4c07-8f68-2d3f83779c80)


# PART 2 : DEEP LEARNING WITH PYTORCH 
For Pytorch deep learning framework we are required to convert our data into tensors.The deep learning is done on these tensors

![Tensors_convert](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/b82a1039-08d1-431c-a135-44d4c42b14f7)

My first deep model was a linear model with 4 inputs and 1 output

![Linear Model](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/d4ffbf08-b63f-4493-8f4e-b4e5f87d631c)

Before we use a model we have to train it and see if it minimizes loss . First I had to do a random prediction before training and then selected my optimizer and loss function

![Selecting optimizer and pretraining](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/b31fbe2f-7e28-4477-884a-8fc4b9ee55ee)

Training the model

![Linear Model training](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/398b875a-1882-48af-a901-56fe3032c5cb)


# RNNS and GRUS 

A recurrent neural network (RNN) is a type of artificial neural network which uses sequential data or time series data.The Gated Recurrent Unit (GRU) is a type of Recurrent Neural Network (RNN) that, in certain cases, has advantages over long short term memory (LSTM). GRU uses less memory and is faster than LSTM, however, LSTM is more accurate when using datasets with longer sequences.Also, GRUs address the vanishing gradient problem . Since we are studying a time sereis ,the first thing was to get the sales values in the sales column
![RNN+GRUS](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/202d3c64-4635-49a2-9393-04c94a1f1d93)

# The sliding window technique

The sliding window technique is a method for iterating over a sequence of data, typically used in the context of machine learning and image processing. It involves dividing the data into overlapping windows of a fixed size, and processing each window independently. I created a sliding window function 
![Sliding window method](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/8c7c7784-eabe-4848-bbfb-8e573e48363c)

I build an RNN network and then performed a forward pass

![RNN](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/078b9aba-fcc8-49e4-9cf3-576738291cb5)

What happens in a RNN ?

RNN works on the principle of saving the output of a particular layer and feeding this back to the input in order to predict the output of the layer. The nodes in different layers of the neural network are compressed to form a single layer of recurrent neural networks.A truncated backpropagation through time neural network is an RNN in which the number of time steps in the input sequence is limited by a truncation of the input sequence. This is useful for recurrent neural networks that are used as sequence-to-sequence models, where the number of steps in the input sequence (or the number of time steps in the input sequence) is greater than the number of steps in the output sequence.
# GRU 

![GRU](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/099ae614-5c40-461e-8952-ceff30f27e96)

What happens in a gru ?
 The basic idea behind GRU is to use gating mechanisms to selectively update the hidden state of the network at each time step. The gating mechanisms are used to control the flow of information in and out of the network .


![gru forward](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/c75d9350-db80-4c55-a607-defe1ace5bf3)



# LSTMS 

LSTM models are a subtype of Recurrent Neural Networks. They are used to recognize patterns in data sequences, such as those that appear in sensor data or  stock prices.

![LSTMs](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/19c9dcf0-2fca-494d-9ea7-f636b88f8ea0)

What happens in a LSTM ?
In an LSTM model, the recurrent weight matrix is replaced by an identify function in the carousel and controlled by a series of gates. The input gate, output gate and forget gate acts like a switch that controls the weights and creates the long term memory function.

![lstm forward](https://github.com/muyale/Walmart-Sales-prediction-using-Pytorch-Neural-Networks/assets/111242297/296dafe5-4b2b-4d3e-a01c-0adf01dd3d61)



# REFERENCES 

https://www.ibm.com/topics/recurrent-neural-networks

https://blog.marketmuse.com/glossary/gated-recurrent-unit-gru-definition/#:~:text=The%20Gated%20Recurrent%20Unit%20(GRU,using%20datasets%20with%20longer%20sequences.






