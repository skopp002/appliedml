In Neural Networks, consider every input parameter (dimension) as beginning layer of the NN
Every input node has an edge to the next layer. For instance if we have x1 and x2 as in this exercise
and we have 3 nodes in the hidden layer (the number of nodes and number of layers are both configurable
and need to be tuned)

Each input node is multiplied with the edge weight and all such products are summed and a bias value is 
added. This is to balance any noise in the system through edge case data points. The output of 1st hidden layer
becomes input for the next layer and there is a mesh of interconnects from the 1st hidden layer to the next as well
and every edge is weighed. At the end, the output is the final layer. In the test phase, the output value 
is compared to the computed final values. The difference in the values (given target value - computed target value)
is taken as error and is fed back into the system. This adjusts the weights on the edges and the process 
is called back propogation. We typically start out with random weights and that is fine since the neural network
trains itself through back propogation to arrive at the right weights. When we have 1 or more hidden layers
they are called deep neural networks.

The method applied on act on the xi * wi of every node (also called the activation method) is the function
applied to this summation at any given layer and each of its nodes. For a sigmoid typically, 

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
Test accuracy: 0.18978102255041582```