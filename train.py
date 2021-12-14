import pandas as pd

df = pd.read_csv("WikiQASent.pos.ans.tsv", sep="\t")
data_set = df[["QuestionID", "Question"]]

import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import string

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For first time
nltk.download("punkt")


def tokenize(sentence):
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    return nltk.word_tokenize(sentence)


stemmer = PorterStemmer()


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for idex, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idex] = 1.0
    return bag


all_words = []
Qtags = []
XY = []

for question in data_set.values:
    Qtags.append(question[0])
    all_words.extend(tokenize(question[1]))
    XY.append((tokenize(question[1]), question[0]))

all_words = [stem(w) for w in all_words]
all_words = sorted(set(all_words))


x_train = []
y_train = []


for sentence, tag in XY:
    bag = bag_of_words(sentence, all_words)
    x_train.append(bag)

    label = Qtags.index(tag)
    y_train.append(label)


x_train = np.array(x_train)
y_train = np.array(y_train)

print(len(y_train) == len(x_train))


class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(y_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


class NerualNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NerualNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


batch_size = 8
hidden_size = 8
output_size = len(Qtags)  # Number of different class
input_size = len(x_train[0])  # Number of the bag of words len(all_words)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NerualNet(input_size, hidden_size, output_size).to(device)


# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch_ in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        # Forward
        outputs = model(words)
        labels = labels.type(torch.LongTensor)
        loss = criterion(outputs, labels)

        # Backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch_ + 1) % 100 == 0:
        print(f"Epoch {epoch_ + 1}/{num_epochs}, loss={loss.item():.4f}")

print(f"Final loss, loss={loss.item():.4f}")

data_ = {"model_state": model.state_dict(), "input_size": input_size, "output_size": output_size, "hidden_size": hidden_size, "all_words": all_words, "tags": Qtags}

FILE = "data.pth"
torch.save(data_, FILE)


print(f"training complete. file saved to {FILE}")
