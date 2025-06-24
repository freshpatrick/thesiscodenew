# Stock Prediction via Deep Attention and Transformer Based Neural Network Combining Mixed Frequency Financial Data and Monthly Revenue

# Architecture diagram
The model used in the experiment of this paper is based on the Transformer model improvement. In the four daily stock price data sets of AAPL, TSLA, MSFT, and IBM, the architecture diagram of the daily stock price data set is as follows:
In the input data section, time_step is set to 10, which means we use the five data features of the previous 10 days, including the daily opening price, highest price, lowest price, trading volume and closing price to predict the closing price of the 11th day, so the input data type is (10,5,1).<br/>  
In the Encoder, Layer Normalization is first performed to make the neurons in each layer approximate a normal distribution to improve the generalization ability of the model. Then the attention mechanism is used to amplify the weight of important features, and then Dropout is used to discard 25% of the neurons to avoid the model's over-reliance on certain features, and then the results of the initial Layer Normalization are added (Add). Then it is divided into two branches to expand the extraction of features of different scales. One branch performs convolution with a kernel size of 1*1, and the number of kernels is set to 2. The other branch performs convolution with a kernel size of 1*3, and the number of kernels is set to 4. Then the outputs of the two convolutional layers are concatenated, and then Dropout is used to discard 25% of the neurons and then enter the convolutional layer with a kernel size of 1*2. In this convolutional layer, the number of kernels is set to the second input data type. Only the dimension value can connect the feature maps together, so the number of kernels is 5. At this point, the Encoder part is completed. <br/>   
Then enter the GAP layer to extract features and prepare to enter the Decoder part. In the Decoder part, the output of the GAP layer is used as the input of the Decoder. It will first go through the fully connected layer of 10 neurons (Dense(10)), then use Dropout to discard 25% of the neurons, and then go through the fully connected layer of 10 neurons (Dense(10)), and then use Dropout. After discarding 25% of the neurons, this output is added to the output of the GAP layer, and the added result is concatenated with the output of the GAP layer, and finally the predicted stock price is output through the fully connected layer (Dense(1)).


![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](https://github.com/freshpatrick/thesiscode/blob/main/image/Architecture.png)

# Environment
- pandas-datareader
* tensorboard
+ keras
+  scikit-learn  
or see the requirements.txt

# How to try
## Download dataset (15 minutes stock、daily stock、StockQM、Astock)
The data set is placed in the data data set. There are 10 data sets in total:
- AAPL
* MSFT
* TSLA
* IBM
* 15 minutes stock(four data)
* StockQM
* Astock


# Operation manual
## Set dataset path
-Our network model is placed in main.py in the network, where relevant model parameters can be selected based on different data sets.The experimental data is placed in the method folder.The following is an introduction to main.py.

### Edit xxx.csv (set path )
#### for example
output_directory = r'xxx\data\daily stock'  
output_path = os.path.join(output_directory, "xxx.csv")  
x_bigdata=np.load(r'xxx\x_bigdata.npy')  
y_bigdata=np.load(r'xxx\y_bigdata.npy')  

# Experimental results
The weight of this experimental data is stored in the checkpoint folder. The experimental data is as shown below
![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](https://github.com/freshpatrick/thesiscode/blob/main/image/Experimentalresults.PNG)
