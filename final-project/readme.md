In Neural Networks, consider every input parameter (dimension) as node in the beginning layer of the NN
Every input node has an edge to the every other node in following/forward layer. For instance here we have 
x1 and x2 as input layer nodes. We could have nodes in the hidden layer (the number of nodes and number of layers are both configurable
and need to be tuned). When using the built in method, the first layer has 8 units which means 8 neurons. And since 
we have 2 input dimensions with color being the output hence the input_dim on 1st layer is 2. As many model.add we have, 
that many neural network layers are stacked up sequentially. Sequential are simpler models. An alternate, more flexible 
model to Sequential is Functional. In this project, we use Sequential model.

The main method and the for loop in main show the actual computation of the 
weights.Each input node is multiplied with the edge weight and all such products are summed and a bias value is 
added. This is to balance any noise in the system through edge case data points. The output of 1st hidden layer
becomes input for the next layer and there is a mesh of interconnects from the 1st hidden layer to the next as well
and every edge is weighed. At the end, the output is the final layer. In the test phase, the output value 
is compared to the computed final values. The difference in the values (given target value - computed target value)
is taken as error and is fed back into the system. This adjusts the weights on the edges and the process 
is called back propogation. We typically start out with random weights and that is fine since the neural network
trains itself through back propogation to arrive at the right weights. When we have 1 or more hidden layers
they are called deep neural networks. 

The method applied to act on the xi * wi of every node (also called the activation method) is the function
applied to this summation at every node of the layer. This is forward propogation.
For a sigmoid typically, 

( 1 / (1 + exp(-x))) would be applied. Where x is a dot product of xi * wi

For given exercise, we do not have a binary classification hence ReLU is more suitable here. 
ReLU has a much simpler computation which is to take the max of 0 and x
max(0, x) 

The derivative or the function applied during back propogation is the difference between computed and actual
output. If the difference is more than 0, it means the error was significant. If not, no correction is made. 

The number of neurons in the output layer would be 3 in our exercise, 
since we have 3 different possible classes - r, g and b


Below is the output from the program

```Using TensorFlow backend.
Weights we computed are  [66693.43435282 34499.43037139]
Now lets try using the built in classifier
WARNING:tensorflow:From /Users/sunitakoppar/PycharmProjects/appliedml/venv/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From /Users/sunitakoppar/PycharmProjects/appliedml/venv/lib/python3.7/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
Epoch 1/1
2019-06-10 17:45:57.406353: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
 32/548 [>.............................] - ETA: 2s - loss: 9.1949 - acc: 0.0625
548/548 [==============================] - 0s 257us/step - loss: 7.3289 - acc: 0.1679
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 8)                 24        
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 27        
=================================================================
Total params: 51
Trainable params: 51
Non-trainable params: 0
_________________________________________________________________
None
Test loss: 5.870483889196911
Test accuracy: 0.18978102255041582



Using both layers with relu activation, the output is as below:
Epoch 1/1
2019-06-11 13:53:37.396956: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
 32/548 [>.............................] - ETA: 1s - loss: 2.2418 - acc: 0.7188
548/548 [==============================] - 0s 243us/step - loss: 3.8827 - acc: 0.5766
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 30)                90        
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 93        
=================================================================
Total params: 183
Trainable params: 183
Non-trainable params: 0
_________________________________________________________________
None
Test loss: 4.123628386615836
Test accuracy: 0.5985401472906127
```


###### This configuration gave the least loss and improved the accuracy:

`self.cnn_model.add(Dense(548, input_dim=2, activation='relu'))            

self.cnn_model.add(Dense(30, input_dim=584, activation='softmax'))

self.cnn_model.add(Dense(3, activation='softmax'))

With Epochs = 300. 1 epoch is an iteration on the given X and y. An epoch of 300 means 
300 iterations. 
`
```Epoch 300/300
 32/548 [>.............................] - ETA: 0s - loss: 0.1614 - acc: 0.9688
548/548 [==============================] - 0s 27us/step - loss: 0.2681 - acc: 0.9234
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 548)               1644      
_________________________________________________________________
dense_2 (Dense)              (None, 30)                16470     
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 93        
=================================================================
Total params: 18,207
Trainable params: 18,207
Non-trainable params: 0
_________________________________________________________________
None
Test loss: 0.22622666915837866
Test accuracy: 0.9416058402861992 
```