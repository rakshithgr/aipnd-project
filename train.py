import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbai
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms, models
import helper as h

import warnings
warnings.filterwarnings("ignore")

# Commanline Arguments Parsing
parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('--save_dir', dest="save_dir")
parser.add_argument('--arch', dest='arch', default='densenet121')
parser.add_argument('--learning_rate', default=0.001, type=float, dest="learning_rate")
parser.add_argument('--hidden_units', default=[512], dest="hidden_units")
parser.add_argument('--epochs', default=30, type=int, dest="epochs")
parser.add_argument('--drop_p', default=0.5, type=float, dest="drop_p")
parser.add_argument('--gpu', action="store_true", default=False, dest="gpu")
args = parser.parse_args()



# adding custom classifier to pretrained model
def add_classifier(model, input_size, output_size, hidden_layer, drop_p):
    model.classifier = h.classifier(input_size, output_size, hidden_layer, drop_p)
    return model


# validation function
def validation(model, criterion, dataloader, device):
    loss = 0
    accuracy = 0

    for images, labels in iter(dataloader):
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        loss += criterion(output, labels).item()

        probs = torch.exp(output)
        equality = (labels.data == probs.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return loss, accuracy


# training model
def train(model, optimizer, criterion, trainloader, validloader, epochs, gpu):

    # setting compute device
    device = "cuda:0" if gpu else "cpu"
    model.to(device)

    # training
    steps = 0
    running_loss = 0
    print_step = 32

    # validation results
    vloss = 0
    vaccuracy = 0

    for e in range(epochs):
        for images, labels in iter(trainloader):
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps%print_step == 0:
                model.eval()

                with torch.no_grad():
                    vloss, vaccuracy = validation(model, criterion, validloader, device)

                print('Epoch: {}/{}\t'.format(e+1, epochs),
                      'Train Loss: {:.3f}\t'.format(running_loss/print_step),
                      'Valid Loss: {:.3f}\t'.format(vloss/len(validloader)),
                      'Valid Accuracy: {:.3f}'.format(vaccuracy/len(validloader)*100))

                running_loss = 0

                model.train()

        # saving model as checkpoint
        model.class_to_idx = class_to_idx
        h.save_model(args.arch, model, optimizer, input_size, output_size, epochs, args.drop_p, args.save_dir, args.learning_rate)



# model training statements
# dataset loading
trainloader, validloader, testloader, class_to_idx = h.load_data(args.data_dir)

# model hyperparameters
input_size = 1024
output_size = 102
hidden_layer = list(map(int, args.hidden_units.strip('[]').split(',')))

# model selection and adding custom classifier
model = h.model_selection(args.arch)
model = add_classifier(model, input_size, output_size, hidden_layer, args.drop_p)

# specifying criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# begin training
train(model, optimizer, criterion, trainloader, validloader, args.epochs, args.gpu)

# print test set accuracy
tloss, taccuracy = validation(model, criterion, testloader, device)
print('\n\n\nTest Set Loss: {:.3f}\nTest Set Accuracy: {:.3f}'.format((tloss/len(testloader)),
                                                                      (taccuracy/len(testloader)*100)))
