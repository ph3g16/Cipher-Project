# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 00:07:07 2018

Program using a neural network to try to find the key for a Bifid Cipher

@author: Peter Hopkinson
"""

from random import shuffle
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import string

""" 1) Load training and evaluation data """

truncated_alphabet = string.ascii_uppercase.replace("J", "")  # "ABCDEFGHIKLMNOPQRSTUVWXYZ"
alphabet_length = len(truncated_alphabet)

Sample_Input = "INCRYPTOGRAPHYACIPHERORCYPHERISANALGORITHMFORPERFORMINGENCRYPTIONORDECRYPTIONASERIESOFWELLDEFINEDSTEPSTHATCANBEFOLLOWEDASAPROCEDUREANALTERNATIVELESSCOMMONTERMISENCIPHERMENTTOENCIPHERORENCODEISTOCONVERTINFORMATIONINTOCIPHERORCODEINCOMMONPARLANCECIPHERISSYNONYMOUSWITHCODEASTHEYAREBOTHASETOFSTEPSTHATENCRYPTAMESSAGEHOWEVERTHECONCEPTSAREDISTINCTINCRYPTOGRAPHYESPECIALLYCLASSICALCRYPTOGRAPHY"
Sample_Output = "UHCRTKSPCBAPHYMHIPFOSEYNTKFORITTITXVSLNAAYHETENLHETVUHBKVSSVHWNPNOYIVQSVHWNPITRORINQTLLPZUBPFIUOCHRPHQTHHADTVRILOLQELPDATTHLYQVWNUIWITQINLITRAUKLESSCOAGRQRPTVNHUQBTAQNLFKNTSPUQBTAQNLSLUQCOBPNHSPCOUYNLRAIRSLADRARQUHSPBTAQNLSLCOBPUHCOAGRQAWOUFHBOBTAQNLNHSYNONYHKNNLATHCOBPHHTHNXHFVLSWATROSPHNRPHQTHHAUQCRTKTAFKSSMMIQOWUXNLTHVQRQBOHWTTREBASTUHCTUHCRTKSPCBAPHYNQFZBTPFQVKRHHRTDTXNSVHWYXTIAQY"
Sample_Mode = 2

Input_Text, Output_Text, Mode = Sample_Input, Sample_Output, Sample_Mode

def clip_text(input_text=Sample_Input, output_text=Sample_Output, mode=Sample_Mode):
    input_length = len(input_text)
    output_length = len(output_text)
    num_blocks = min(math.floor(input_length/mode), math.floor(output_length/mode))    
    return(input_text[:mode*num_blocks], output_text[:mode*num_blocks], num_blocks)
### end clip_text

class DataPoint:
    # class DataPoint associates an input with an output (in this case an input letter pair with an output letter pair)
    # using a class structure makes for easy shuffling
    def __init__(self, input_string, output_string, mode, block_number):
        self.input_data = [(truncated_alphabet.index(x)+0.5)/alphabet_length for x in input_string[block_number:block_number + mode]]
        self.output_data = [truncated_alphabet.index(x) for x in output_string[block_number:block_number + mode]]
### end DataPoint

train_input, train_output, train_size = clip_text(Sample_Input, Sample_Output, Sample_Mode)
training_set = [DataPoint(train_input, train_output, Sample_Mode, x) for x in range(train_size)]
batch_size = train_size # could experiment with other values, e.g. 1, train_size, etc

""" 2) Define network structure and initialise session """

# placeholder variables
plaintext = tf.placeholder(tf.float32, [None, Sample_Mode]) # Number of examples, number of inputs
ciphertext = tf.placeholder(tf.float32, [None, Sample_Mode]) # Number of examples, number of outputs

# network layers
with slim.arg_scope([slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.2),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    # process plaintext
    layer_1_a = slim.fully_connected(plaintext, 25, scope='layer_1', reuse=None)
    layer_2_a = slim.fully_connected(layer_1_a, 25, scope='layer_2', reuse=None)
    output_a = slim.fully_connected(layer_2_a, 2*Sample_Mode, scope='output', reuse=None)
    output_a = tf.reshape(output_a, [batch_size, Sample_Mode, 2])
    output_a = tf.transpose(output_a, perm=[0,2,1])
    output_a = tf.reshape(output_a, [batch_size, 2*Sample_Mode])
    # process ciphertext
    layer_1_b = slim.fully_connected(ciphertext, 25, scope='layer_1', reuse=True)
    layer_2_b = slim.fully_connected(layer_1_b, 25, scope='layer_2', reuse=True)
    output_b = slim.fully_connected(layer_2_b, 2*Sample_Mode, scope='output', reuse=True)
    
comparison = tf.reduce_sum(tf.abs(output_a - output_b))

# backpropogation variables
cross_entropy = comparison
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)
# mid-training evaluation test
a_rounded = tf.floor(output_a*5)/5
b_rounded = tf.floor(output_b*5)/5
check_predictions = tf.equal(a_rounded, b_rounded)
error = tf.reduce_mean(tf.cast(check_predictions, tf.float32))

# initialize session
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
print("Neural net intialised, sample data loaded.")

""" 3) Train network """

def train_network(num_epochs=2000):
    # run training
    no_of_batches = int(train_size / batch_size)
    accuracy = sess.run(cross_entropy, {plaintext: [element.input_data for element in training_set][:batch_size], ciphertext: [element.output_data for element in training_set][:batch_size]})
    print("Accuracy ", accuracy)
    for epoch in range(num_epochs): # start epoch
        shuffle(training_set)
        ptr = 0
        for j in range(no_of_batches): # cycle through all batches within training data
            batch = training_set[ptr:ptr+batch_size]
            inp = [element.input_data for element in batch]
            out = [element.output_data for element in batch]
            ptr+=batch_size
            sess.run(minimize, {plaintext: inp, ciphertext: out}) # run backpropogation
        if epoch-(math.floor(epoch/10)*10) is 9: print("Epoch ",str(epoch+1)) # print every 10th epoch (the system actually labels these as the ###9th epoch since it treats the first as epoch zero)
        if epoch-(math.floor(epoch/100)*100) is 99:
            accuracy = sess.run(error, {plaintext: [element.input_data for element in training_set][:batch_size], ciphertext: [element.output_data for element in training_set][:batch_size]})
            loss = sess.run(cross_entropy, {plaintext: [element.input_data for element in training_set][:batch_size], ciphertext: [element.output_data for element in training_set][:batch_size]})
            print("Accuracy: ", accuracy, ", Loss: ", loss)
### end train_network
    
""" 4) Evaluate model, this stage is essentially reading the answer rather than actually confirming that the network has produced a viable solution """

def eval_network():

    # reveal polybius information
    eval_alphabet = truncated_alphabet*math.ceil(Sample_Mode*batch_size/alphabet_length)
 
    eval_input, eval_output, eval_size = clip_text(eval_alphabet, eval_alphabet, Sample_Mode)
    eval_set = [DataPoint(eval_input, eval_output, Sample_Mode, x) for x in range(eval_size)]
    
    coordinates = sess.run(b_rounded, {plaintext: [element.input_data for element in eval_set][:batch_size], ciphertext: [element.output_data for element in eval_set][:batch_size]}) 
    print(coordinates)
### end eval_network

""" 5) User interface """
    
def main():
    # this function is run if the module is opened as a standalone module
    print("Main menu. Please choose from the following options:")
    print("1) Enter new input string")
    print("2) Enter new output string")
    print("3) Select mode")
    print("4) Display current settings")
    print("5) Train model")
    print("6) Evaluate model")
    print("7) Quit")
    user_input = input("Enter your choice: ")
    if user_input == str(1):
        Input_Text = input("Enter new input string:")
    elif user_input == str(2):
        Output_Text = input("Enter new output string:")
    elif user_input == str(3):
        new_mode = input("Enter an integer value to represent the new mode:")
        if isinstance(new_mode, int) and new_mode > 0:
            Mode = new_mode
        else: print("Invalid entry, mode not updated. Mode needs to be a positive integer.")
    elif user_input == str(4):
        global Input_Text, Output_Text, Mode
        print("Input String: " + Input_Text)
        print ("Output String: " + Output_Text)
        print("Mode: {}".format(Mode))
    elif user_input == str(5):
        train_network()
    elif user_input == str(6):
        eval_network()
    elif user_input == str(7):
        return
    else: print("Menu option not recognised. You need to enter an integer value corresponding to the appropriate option.")
    # after completing the menu option we want to return to the main menu, notice that this doesn't happen if the user elected to quit    
    main()
### end main

if __name__ == "__main__":
    # checks if the program is being used as a standalone module (rather than being imported by another module)
    # if in standalone mode then run the main method
    main()
