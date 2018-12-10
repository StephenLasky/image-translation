import torch
import torch.nn as nn
import data

LEARNING_RATE = 0.0005

BATCH_SIZE = 32
TRAIN_SET_BATCHES = 1
TRAIN_SET_SIZE = BATCH_SIZE * TRAIN_SET_BATCHES

NUM_EPOCHS = 500

class Model1(nn.Module):
    def __init__(self, num_hidden, hidden_width):
        super(Model1, self).__init__()
        self.relu = nn.ReLU()

        self.hiddenLayers = []
        self.inputLayer = nn.Linear(1536, hidden_width)
        self.outputLayer = nn.Linear(hidden_width, 1536)

        for i in range(num_hidden):
            self.hiddenLayers.append(nn.Linear(hidden_width, hidden_width))

    def forward(self, x):
        out = self.inputLayer(x)
        out = self.relu(out)

        for layer in self.hiddenLayers:
            out = layer(out)
            out = self.relu(out)

        out = self.outputLayer(out)

        return out

m = Model1(1,1536)
lossFunction = nn.L1Loss()
optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)

# put together the train set
raw = data.get_im_set(TRAIN_SET_SIZE, type=0)
train_x = []
train_y = []
for i in range(TRAIN_SET_SIZE):
    z = data.split_vec(raw[i])
    train_x.append(z[0])
    train_y.append(z[1])

train_x = data.vecs_to_tensors(train_x)
train_y = data.vecs_to_tensors(train_y)

y_predictions = []

for epoch in range(NUM_EPOCHS):
    if epoch != 0 and epoch % 50 == 0:
        LEARNING_RATE /= 2
        # optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)

    for i in range(TRAIN_SET_BATCHES):
        x = train_x
        y = train_y

        y_pred = m.forward(x)
        loss = lossFunction(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_predictions.append(y_pred)

    print("epoch {} done".format(epoch))

v = data.tensors_to_vecs(y_predictions[0])
print("{} {} {}".format(len(y_predictions), len(y_predictions[0]), len(y_predictions[0][0])))
print("{} {} {}".format(len(v), len(v[0]), 1))

train_x_vecs = data.tensors_to_vecs(train_x)
for i in range(0,len(y_predictions), 49):
    v = data.tensors_to_vecs(y_predictions[i])
    for j in range(len(v)):
        v[j] = data.comb_vec(train_x_vecs[j], v[j])
        v[j] = data.cifar_im_to_std_im(v[j], 32, 32)

    data.print_img(data.combine_img(v))
