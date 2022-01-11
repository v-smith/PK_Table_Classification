import torch
from torch import nn
from torch.nn import functional as F, Sequential

# ========== Fully connected neural network with 2 hidden layers and embeddings ============#

class NeuralNet(nn.Module):
    def __init__(self, num_classes, hidden_size, embeds_size, vocab_size, padding_idx, drop_out):
        super(NeuralNet, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embeds_size, padding_idx=padding_idx)
        #self.l1 = nn.Linear(50000, hidden_size)
        self.l1 = nn.Linear(embeds_size, hidden_size)  # or number of classes if only one layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)  # defines proportion of neurons to drop out
        self.l2 = nn.Linear(hidden_size, num_classes) #(hidden_size + vocab_size)

    def forward(self, x):  # multi
        embeddings = self.embedding(x)
        #unbound_rows = [torch.unbind(row) for row in embeddings]
        #concat = [torch.cat(row) for row in unbound_rows]
        #stacked = torch.stack(concat)
        max_pooled = torch.max(embeddings, dim=1, keepdim=False)[0]
        # out_dropout = self.dropout(max_pooled)
        out_l1 = self.l1(max_pooled)
        #out_l1 = self.l1(stacked)
        out_relu = self.relu(out_l1)
        out_dropout = self.dropout(out_relu)
        #concat_multi = torch.cat((multi, out_dropout), dim=1)
        out_l2 = self.l2(out_dropout) #concat_multi
        return out_l2


# ========== Fully connected neural network with 1 hidden layers and BOW ============#
class BOW_NeuralNet(nn.Module):
    def __init__(self, num_classes, hidden_size, input_size, drop_out):
        super(BOW_NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, num_classes)  # hidden size or number of classes if only one layer
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(drop_out)  # define proportion of neurons to drop out
        # self.double() #if feeding in a Torch.Double type

    def forward(self, x):
        x = x.float()
        # max_pooled = torch.max(x, dim=1, keepdim=False)[0].float()
        out_dropout = self.dropout(x)
        out_l1 = self.l1(out_dropout)
        out_relu = self.relu(out_l1)
        # out_l2 = self.l2(out_relu)
        return out_relu


# ========== CNN- NGRAM CLASS with Embeddings ============#

class CNN(nn.Module):

    def __init__(self, num_classes, input_channels, num_filters, filter_sizes, embeds_size, vocab_size,
                 padding_idx, stride, drop_out):
        super(CNN, self).__init__()
        """
        n.b this is an ngram of words architecture, meaning different kernel sizes are applied to the same table and each of these outputs will then be reduced with max pooling 
        output_size : 9 = (classes)
        input_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
        filter_sizes : A list consisting of different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
        """
        self.embedding = torch.nn.Embedding(vocab_size, embeds_size, padding_idx=padding_idx)
        self.output_size = num_classes
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes  # number of sub-words at a time
        self.stride = stride  # Number of strides for each convolution, default is 1

        # conv layers definition
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(self.input_channels, self.num_filters, (K, embeds_size), self.stride) for K in
             self.filter_sizes])
        # self.conv2 = nn.Conv1d(self.input_channels, self.out_channels, (self.kernel_heights[1], embeds_size), self.stride)
        self.dropout = nn.Dropout(drop_out)
        self.fc1 = nn.Linear((len(filter_sizes) * num_filters), num_classes)
        #self.fc1 = nn.Linear((len(filter_sizes) * num_filters) + vocab_size, num_classes)

    def forward(self, x): #multi
        embeds = self.embedding(x)
        input = embeds.unsqueeze(1)
        convs = [F.relu(conv(input)).squeeze(3) for conv in self.convs1]
        pooled = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in convs]
        concat = torch.cat(pooled, 1)
        concat = self.dropout(concat)
        #concat = torch.cat((concat, multi), dim=1)
        logits = self.fc1(concat)
        return logits


# ========== CNN- SEQUENTIAL CLASS with Embeddings ============#

class CNN_Seq(nn.Module):

    def __init__(self, num_classes, input_channels, num_filters, filter_sizes, embeds_size, vocab_size,
                 padding_idx, stride, drop_out):
        super(CNN_Seq, self).__init__()

        self.embedding = torch.nn.Embedding(vocab_size, embeds_size, padding_idx=padding_idx)
        self.output_size = num_classes
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.stride = stride
        self.dropout = drop_out
        self.embeds_size = embeds_size

        self.cnn1_layers = Sequential(
            # first conv layer
            nn.Conv2d(self.input_channels, 100, kernel_size=(5, 5), stride=5, padding=1),
            nn.ReLU(inplace=True))

        self.maxpool1 = Sequential(
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Dropout(self.dropout))

        self.cnn2 = Sequential(
            nn.Conv2d(100, 200, kernel_size=(5, 1), stride=5, padding=1),
            nn.ReLU(inplace=True))

        self.maxpool2 = Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout))

        self.linear_layers = Sequential(nn.Linear(in_features=1000, out_features=200),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(in_features=200, out_features=100),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(in_features=100, out_features=num_classes))

    def forward(self, x):
        embeds = self.embedding(x)
        input = embeds.unsqueeze(1)
        conv1 = self.cnn1_layers(input)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.cnn2(maxpool1)
        output_convs = self.maxpool2(conv2)
        adjust = output_convs.view(output_convs.size(0), -1)
        final_output = self.linear_layers(adjust)
        return final_output
