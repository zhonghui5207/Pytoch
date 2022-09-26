"""
Examples::
>> > rnn = nn.RNNCell(10, 20)
>> > input = torch.randn(6, 3, 10)
>> > hx = torch.randn(3, 20)
>> > output = []
>> > for i in range(6):
    hx = rnn(input[i], hx)
    output.append(hx)
"""

import torch
import torch.nn as nn
import numpy as np

batch_size = 2
input_size = 4
hidden_size = 4
seq_len = 3
number_layers = 1

inputs = torch.randn(seq_len, batch_size, input_size)
print("inputs.shape,inputs", (inputs.shape, inputs))
hx = torch.randn(number_layers,batch_size, hidden_size)
print("hx.shape,hx", (hx.shape, hx))
# output = []
# rnn = nn.RNNCell(input_size,hidden_size)
#
# for i in range(seq_len):
#     hx = rnn(inputs[i],hx)
#     print("i,hx:",(i,hx))
#     output.append(hx)
# print(output)


rnn1 = nn.RNN(input_size=input_size, hidden_size=hidden_size,num_layers=number_layers)
"""

    Examples::

        >>> rnn = nn.RNN(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
"""

out, hx1 = rnn1(inputs, hx)
print("out.shape,out", (out.shape, out))
print("hx1.shape,hx", (hx1.shape, hx1))
