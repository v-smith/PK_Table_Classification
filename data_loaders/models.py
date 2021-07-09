import torch
from torch import nn

#========== Fully connected neural network with 2 hidden layers ============#
#
class NeuralNet(nn.Module):
    def __init__(self, num_classes, hidden_size, embeds_size, vocab_size, padding_idx):
        super(NeuralNet, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embeds_size, padding_idx=padding_idx)
        self.l1 = nn.Linear(embeds_size, hidden_size)  # or number of classes if only one layer
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.25)  # define proportion of neurons to drop out
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
#2 conv layers

class CNN(nn.Module):

    def __init__(self, num_classes, embeds_size, vocab_size, padding_idx, stride, out_size):
        super(ConvNet, self).__init__()
        #embedding layer definition
        self.embedding = torch.nn.Embedding(vocab_size, embeds_size, padding_idx=padding_idx)

        # Output size for each convolution
        self.out_size = out_size
        # Number of strides for each convolution
        self.stride = stride

        # CNN parameters definition
        # Kernel sizes
        self.kernel_1 = 2
        self.kernel_2 = 3

        #conv layers definition
        self.conv1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_2, self.stride)

        # Max pooling layers definition
        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)

        #fully connected layers
        self.fc = nn.Linear(self.in_features_fc(), num_classes)

        # Dropout definition
        self.dropout = nn.Dropout(0.25)

    def in_features_fc(self):
        '''Calculates the number of output features after Convolution + Max pooling
        Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
        Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
        '''
        # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
        out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)

        # Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
        out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_conv_2 = math.floor(out_conv_2)
        out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_pool_2 = math.floor(out_pool_2)

        # Returns "flattened" vector (input for fully connected layer)
        return (out_pool_1 + out_pool_2) * self.out_size

    def forward(self, x):
        #tokens to embeddings
        embeddings = self.embedding(x)

        #conv layer 1 applied
        conv1= self.conv1(x)
        relu1= torch.relu(conv1)
        pooled1= self.pool_1(relu1)

        #conv layer 2 applied
        conv2 = self.conv_2(x)
        relu2 = torch.relu((conv2))
        pooled2 = self.pool_2(relu2)

        # The output of each convolutional layer is concatenated into a unique vector
        union = torch.cat((pooled1, pooled2), 2) #check this
        union = union.reshape(union.size(0), -1)

        # The "flattened" vector is passed through a fully connected layer
        out = self.fc(union)
        # Dropout is applied
        ##out = self.dropout(out)

        return out.squeeze()


#========== RNN CLASS ============#
