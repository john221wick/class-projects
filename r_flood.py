import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv("Analyze_Flood_Points._with_address.csv")

x = df.drop(['Flood', 'state', 'city'], axis = 1).values.astype(np.float32)
y = df['Flood'].values.astype(np.float32)

x_tensor = torch.tensor(x)
y_tensor = torch.tensor(y).view(-1, 1)

mean = x_tensor.mean(0, keepdim = True)
std = x_tensor.std(0, unbiased = False, keepdim = True)
std[std == 0] = 1
x_tensor = (x_tensor - mean )/std

dataset = TensorDataset(x_tensor, y_tensor)
# what do you mean 

test_size = int(len(dataset)*0.2)
train_size = int(len(dataset)) - test_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = 64, shuffle = True)

class FloodPredictor(nn.Module):
    def __init__(self, num_features):
        super(FloodPredictor, self).__init__()
        self.layer1 = nn.Linear(num_features, 1800)
        self.layer2 = nn.Linear(1800, 1164)
        self.output_layer = nn.Linear(1164, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output_layer(x))
        return x

model = FloodPredictor(15)

optimizer = optim.Adam(model.parameters(), lr = 0.001)

def train(model, optimizer, train_loader, epoch):
	model.train()
	for epoch in range(epoch):
		for inputs, labels in train_loader:
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = nn.BCELoss()(outputs, labels)
			loss.backward()
			optimizer.step()

		if epoch%10 == 0:
			print(f'Epoch {epoch+1}/{epoch}, Loss: {loss.item()}')

train(model, optimizer, train_loader, 100)

def test(model, test_loader):
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for inputs, labels in test_loader:
			outputs = model(inputs)
			predicted = outputs.round()
			correct += (predicted.view(-1) == labels.view(-1)).float().sum().item()
			total += labels.size(0)
		accuracy = 100*correct/total
		print(f'accuracy : {accuracy}')

test(model, test_loader)






