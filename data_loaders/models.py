import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

#========== Fully connected neural network with 2 hidden layers ============#
#
class NeuralNet(nn.Module):
    def __init__(self, num_classes, hidden_size, embeds_size, vocab_size, padding_idx):
        super(NeuralNet, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embeds_size, padding_idx=padding_idx)
        self.l1 = nn.Linear(embeds_size, hidden_size)  # or number of classes if only one layer
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.4)  # define proportion of neurons to drop out
        # self.double() #if feeding in a Torch.Double type

    def forward(self, x):
        embeddings = self.embedding(x)
        max_pooled = torch.max(embeddings, dim=1, keepdim=False)[0]
        out_dropout = self.dropout(max_pooled)
        out_l1 = self.l1(out_dropout)
        out_relu = self.relu(out_l1)
        out_l2 = self.l2(out_relu)
        # no activation and no softmax at the end
        return out_l2

#========== CNN CLASS ============#

class CNN(nn.Module):

    def __init__(self, seq_len, num_classes, input_channels, out_channels, kernel_heights, embeds_size, vocab_size, padding_idx, stride):
        super(CNN, self).__init__()
        """
        output_size : 9 = (classes)
        input_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
        out_channels : Number of output channels after convolution operation performed on the input matrix
        kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
        vocab_size : Size of the vocabulary containing unique words
        embedding_size : Embedding dimension of  word embeddings
        seq_len: input size (length of input sequence) 
        """
        self.embedding = torch.nn.Embedding(vocab_size, embeds_size, padding_idx=padding_idx)
        self.output_size = num_classes
        self.input_channels = input_channels
        self.out_channels = out_channels
        #number of words
        self.kernel_heights = kernel_heights
        # Number of strides for each convolution
        self.stride = stride

        #conv layers definition
        self.conv1 = nn.Conv1d(self.input_channels, self.out_channels, (self.kernel_heights[0], embeds_size), self.stride)
        self.conv2 = nn.Conv1d(self.input_channels, self.out_channels, (self.kernel_heights[1], embeds_size), self.stride)
        self.conv3 = nn.Conv1d(self.input_channels, self.out_channels, (self.kernel_heights[2], embeds_size), self.stride)
        # Dropout
        self.dropout = nn.Dropout(0.25)
        # fully connected layer
        self.fc = nn.Linear(len(kernel_heights) * out_channels, num_classes)

    def conv_block(self, tmp_input, conv_layer):
        conv_out = conv_layer(tmp_input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        #try these
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        # maxpool_out.size() = (batch_size, out_channels)
        #max pool across one dimension (all embeddings can be selected)
        #average pool
        return max_out

    def forward(self, x):
        #tokens to embeddings
        input = self.embedding(x)
        #input.size()= (batch_size, num_seq, embedding length)
        input = input.unsqueeze(1)
        #conv layer 1
        conv1= self.conv_block(input, self.conv1)
        conv2= self.conv_block(input, self.conv2)
        conv3= self.conv_block(input, self.conv3)

        all_out = torch.cat((conv1, conv2, conv3), 1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)
        # fc_in.size()) = (batch_size, num_kernels*out_channels)
        logits = self.fc(fc_in) #error on this line

        return logits

