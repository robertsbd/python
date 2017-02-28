import random # get the random number generator
import numpy as np # for array and matrix operations
import math

class Network:
    """ docstring contains calculations across the entire network, such as error calculations and backprop algorithms """

    # init function
    def __init__(self, name, learning_rate):
        self.name = name
        self.learning_rate = learning_rate

    # get the learning set
    def training_set(self, inputs, outputs):
        """ get training inputs and outs - for the moment just start with a single input and output array """        
        self.training_input = inputs
        self.training_output = outputs

    ############# error calculation functions #######
    def calc_error_output(self, output_layer, expected): # get error for output nodes

        error_output_layer = lambda expected, output: (expected - output) * self.transfer_derivative(output) # define error calc as lambda function

        error = NodeLayer(output_layer.nodes.shape[0])

        for i in range(0, output_layer.nodes.shape[0]):
            error.error[i] = error_output_layer(expected[i], output_layer.nodes[i])
        return error

    def calc_error_hidden(self, layer, error_layer, connect): # layer is the current layer, the error layer is the layer below that is being fed in backwards

        error_notoutput_layer = lambda connection, errornode, output: (connection * errornode) * self.transfer_derivative(output)  # define error calc as lambda function

        error = NodeLayer(layer.nodes.shape[0])

        for i in range(0, layer.nodes.shape[0]):
            for j in range(0, error_layer.nodes.shape[0]):
                error.error[i] += error_notoutput_layer(connect.connections[i,j], error_layer.nodes[j], layer.nodes[i])
        return error

    transfer_derivative = lambda self, output: output * (1.0 - output) 
    """ docstring calculate the derivative of a nodes output """

class NodeLayer:
    """ docstring NodeLayer contains a 1D array of node values
        instance variables
        self.nodes <- 1D array of node values """    
    def __init__(self, n):
        """ doctring __init__() contains a list of all the nodes and the error associated with each node for training """
        self.nodes = np.zeros(n)
        self.error = np.zeros(n)

class Connections:

    """ docstring Connections contains the wieghts of the connections between two layers, also contains method on those weights
    instance variables 
    self.connections <- a 2D array containing the weights of the connections between two layers, 
                        index of first dimension is connection to input layer node 
                        index of second dimension to output layer node """

    ############# functions for initialising wieghts ##############
    def __init__(self, i, j): 
        """ docstring __init__() connections as 2D array of zeros, i = num of input nodes, j = num of output nodes """
        self.connections = np.zeros((i, j))

    def gen_random(self): 
        """ docstring gen_random() will put random numbers in all values of self.connections """
        for i in range(0, self.connections.shape[0]):
            for j in range(0, self.connections.shape[1]):
                self.connections[i, j] = random.random()

    ############# functions for forward propogation ###############
    def forward_propogation(self, input_layer): 
        """ docstring forward_propogation() will forward propogate from an input_layer and return an output layer
        input_layer of type NodeLayer
        output_layer of type NodeLayer """
        output_layer = NodeLayer(self.connections.shape[1]) # output layer is size of the second dimension of the connection matrix

        for i in range(0, input_layer.nodes.shape[0]):
            for j in range(0, output_layer.nodes.shape[0]):       
                output_layer.nodes[j] = self.activation(self.integration(input_layer, j))

        return output_layer

    def integration(self, input_layer, j):
        """ docstring integration sums together input_layer (type NodeLayer) * connections to output_layer, returns single numeric value """
        integration = 0

        for i in range(0,input_layer.nodes.shape[0]):
            integration += input_layer.nodes[i] * self.connections[i, j]
        
        return integration

    def activation(self, input_activation): 
        """ docstring activation function takes input_activation (type numeric) returns numeric"""
        return 1 / (1 + math.exp(-input_activation)) # sigmoid activation function

    ########### functions for updating wieghts after error calculation ##############

    def update_connection_weights(self, learning_rate, input_layer, output_layer):
        
        for i in range(0, input_layer.nodes.shape[0]):
            for j in range(0, output_layer.nodes.shape[0]):       
                self.connections[i, j] += learning_rate * output_layer.error[j] * input_layer.nodes[i]

# Define out network
my_network = Network("3 layer NN", 1)
# set up our nodes
input_layer = NodeLayer(2)
hidden_layer = NodeLayer(3)
output_layer = NodeLayer(1)
# set up our connections and then give them random values
connect_in_hid = Connections(2,3) # create first layer of weights
connect_hid_out = Connections(3,1) # create first layer of weights
# set two sets of connections to random
connect_in_hid.gen_random()
connect_hid_out.gen_random()


# define our training sequence
my_network.training_set(np.array([0.5, 0.25]), np.array([1.0]))

input_layer.nodes = my_network.training_input

output_list = [] # list to store outputs
error_list = [] # this is a list to store the errors so we can watch them decrease

for i in range(0,5):
    hidden_layer = connect_in_hid.forward_propogation(input_layer)
    output_layer = connect_hid_out.forward_propogation(hidden_layer)

    print "ITERATION: ", i, "\n"

    # print out the network
    print "INPUT LAYER:\n", input_layer.nodes
    print "CONNECTIONS INPUT->HIDDEN:\n", connect_in_hid.connections
    print "HIDDEN LAYER:\n", hidden_layer.nodes 
    print "CONNECTION HIDDEN->OUTPUT:\n", connect_hid_out.connections 
    print "OUTPUT LAYER\n", output_layer.nodes, "\n\n"

    #calculate the error associated with each node
    output_layer.error = my_network.calc_error_output(output_layer,my_network.training_output).error
    hidden_layer.error = my_network.calc_error_hidden(hidden_layer, output_layer, connect_hid_out).error

    # update the weights
    connect_in_hid.update_connection_weights(my_network.learning_rate, input_layer, hidden_layer)
    connect_hid_out.update_connection_weights(my_network.learning_rate, hidden_layer, output_layer)

    #return errors
    print "HIDDEN ERROR", hidden_layer.error
    print "OUTPUT ERROR", output_layer.error, "\n\n"
    error_list.append(output_layer.error[0])
    output_list.append(output_layer.nodes[0])
# print out the entire error

print "I: ERROR, OUTPUT"
for i in range(0, len(error_list)):
    print i, ": ", error_list[i], ", ", output_list[i]

print "\n\nINPUT LAYER:\n", input_layer.nodes
print "TARGET OUTPUT\n", my_network.training_output