import pyconll
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.autograd as autograd
from tqdm import tqdm
import pickle, time

from model.lstm_char import LSTMCharTagger
from util import make_ixs
from dataset import PerseusCoNLLUDataset

torch.manual_seed(1)

WORD_EMBEDDING_DIM = 10
CHAR_EMBEDDING_DIM = 5
HIDDEN_DIM = 100
CHAR_REPR_DIM = 50
BATCH_SIZE = 1
NUM_EPOCHS = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train():
    # Read input
    train = PerseusCoNLLUDataset(pyconll.load_from_file(
        "data/perseus-conllu/grc_perseus-ud-train.conllu"
    ))
    val = PerseusCoNLLUDataset(pyconll.load_from_file("data/perseus-conllu/grc_perseus-ud-dev.conllu"))
    word_to_ix, char_to_ix, tag_to_ix = train.get_indices()

    # Initialize model
    model = LSTMCharTagger(
        WORD_EMBEDDING_DIM,
        CHAR_EMBEDDING_DIM,
        CHAR_REPR_DIM,
        HIDDEN_DIM,
        len(word_to_ix),
        len(char_to_ix),
        len(tag_to_ix),
        1,
        device,
    )

    model.to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []

    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        correct = 0
        total = 0

        model.train()
        for words, tags in tqdm(train, desc='Training...'):
            word_characters_ixs = {}
            for word in words:
                word_ix = (
                    torch.tensor([word_to_ix[word]]).to(device)
                    if word in word_to_ix
                    else torch.tensor([word_to_ix["<UNK>"]]).to(device)
                )
                char_ixs = make_ixs(word, char_to_ix, device)
                word_characters_ixs[word_ix] = char_ixs

            targets = make_ixs(tags, tag_to_ix, device)

            model.zero_grad()

            model.init_word_hidden()
            tag_scores = model(word_characters_ixs)

            loss = loss_function(tag_scores, targets)
            loss.backward()  # calculate gradients
            optimizer.step()  # update hidden layers based on gradients

            train_loss += loss
            _, predicted = tag_scores.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = train_loss / len(train)
        train_accuracy = 100.0 * correct / total

        print("Epoch %d, training loss = %.4f, training accuracy = %.2f%%" % (epoch + 1, train_loss / len(train), train_accuracy))
        train_losses.append(train_loss / len(train))

        # Evaluate on validation dataset
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        for words, tags in tqdm(val, desc='Evaluating...'):
            word_characters_ixs = {}
            for word in words:
                word_ix = (
                    torch.tensor([word_to_ix[word]]).to(device)
                    if word in word_to_ix
                    else torch.tensor([word_to_ix["<UNK>"]]).to(device)
                )
                char_ixs = make_ixs(word, char_to_ix, device)
                word_characters_ixs[word_ix] = char_ixs

            targets = make_ixs(tags, tag_to_ix, device)

            model.zero_grad()

            model.init_word_hidden()
            tag_scores = model(word_characters_ixs)
            loss = loss_function(tag_scores, targets)

            val_loss += loss.item()
            _, predicted = tag_scores.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        val_accuracy = 100.0 * correct / total

        print("Epoch %d, validation accuracy = %.2f%%" % (epoch + 1, val_accuracy))

        with open("model-%d.pickle" % time.time(), "wb") as f:
            pickle.dump(model, f)


if __name__ == "__main__":
    train()
