from multiprocessing import cpu_count

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from deepmorpheus.util import add_element_wise


class LSTMCharTagger(pl.LightningModule):
    """LSTM part-os-speech (PoS) tagger."""

    def __init__(self, hparams, vocab, train_data=None, val_data=None):
        super(LSTMCharTagger, self).__init__()
        self.hparams = hparams
        self.train_data = train_data
        self.val_data = val_data
        self.vocab = vocab

        self.single_output = False
        self.directions = 1 if self.hparams.disable_bidirectional else 2
        self.hparams.num_lstm_layers = 2

        self.word_embeddings = nn.Embedding(len(self.vocab.words), self.hparams.word_embedding_dim)
        self.word_lstm_input_dim = self.hparams.word_embedding_dim if hparams.disable_char_level else self.hparams.word_embedding_dim + self.hparams.char_lstm_hidden_dim * self.directions
        self.word_lstm = nn.LSTM(
            self.word_lstm_input_dim,
            self.hparams.word_lstm_hidden_dim,
            bidirectional=self.directions > 1,
            dropout=self.hparams.dropout,
            num_layers=self.hparams.num_lstm_layers
        )
        self.init_word_hidden()

        if not hparams.disable_char_level:
            self.char_embeddings = nn.Embedding(len(self.vocab.chars), self.hparams.char_embedding_dim)
            self.char_lstm = nn.LSTM(
                self.hparams.char_embedding_dim,
                self.hparams.char_lstm_hidden_dim,
                bidirectional=self.directions > 1,
                dropout=self.hparams.dropout,
                num_layers=self.hparams.num_lstm_layers
            )
            self.init_char_hidden()

        tag_fc = {}
        for idx in range(len(self.vocab.tags) if not self.single_output else 1):
            num_tag_outputs = len(self.vocab.tags[idx])
            fc = nn.Linear(self.hparams.word_lstm_hidden_dim * self.directions, num_tag_outputs)
            tag_fc[self.vocab.tag_names[idx]] = fc
        
        self.tag_fc = nn.ModuleDict(tag_fc)
        self.tag_len = len(self.tag_fc)


    def init_word_hidden(self):
        """Initialise word LSTM hidden state."""
        self.word_lstm_hidden = (
            torch.zeros(self.directions * self.hparams.num_lstm_layers, 1, self.hparams.word_lstm_hidden_dim).to(self.device),
            torch.zeros(self.directions * self.hparams.num_lstm_layers, 1, self.hparams.word_lstm_hidden_dim).to(self.device),
        )

    def init_char_hidden(self):
        """Initialise char LSTM hidden state."""
        self.char_lstm_hidden = (
            torch.zeros(self.directions * self.hparams.num_lstm_layers, 1, self.hparams.char_lstm_hidden_dim).to(self.device),
            torch.zeros(self.directions * self.hparams.num_lstm_layers, 1, self.hparams.char_lstm_hidden_dim).to(self.device),
        )

    def forward(self, sentence):
        """The main forward function, this does the actual heavy lifting"""
        words = torch.tensor([word for word, _, _ in sentence]).to(self.device)

        word_embeddings = self.word_embeddings(words)
        word_embeddings_bs = word_embeddings.view(len(sentence), self.hparams.batch_size, self.hparams.word_embedding_dim)

        word_repr = []
        if not self.hparams.disable_char_level:
            for word_idx in range(len(sentence)):
                self.init_char_hidden() # Don't store representation between words
                chars = torch.tensor(sentence[word_idx][1]).to(self.device)

                # Character-level representation is the LSTM output of the last character.
                chars_repr = None
                for char in chars:
                    char_embed = self.char_embeddings(char)
                    chars_repr, self.char_lstm_hidden = self.char_lstm(
                        char_embed.view(1, self.hparams.batch_size, self.hparams.char_embedding_dim), self.char_lstm_hidden
                    )

                chars_repr = chars_repr.view(1, self.hparams.char_lstm_hidden_dim * self.directions)

                word_repr.append(word_embeddings[0].unsqueeze(0))
                word_repr.append(chars_repr)
        else:
            for word_idx in range(len(sentence)):
                word_repr.append(word_embeddings[0].unsqueeze(0))

        # Each sentence embedding dimensions are word embedding dimensions + character representation dimensions
        word_repr = torch.cat(word_repr, dim=1) # From row to column

        sentence_repr, self.word_lstm_hidden = self.word_lstm(
            word_repr.view(len(sentence), self.hparams.batch_size, self.word_lstm_input_dim),
            self.word_lstm_hidden,
        )

        sentence_repr = sentence_repr.view(len(sentence), self.hparams.word_lstm_hidden_dim * self.directions)

        all_word_scores = [[] for _ in range(len(sentence))]

        for tag_name in self.vocab.tag_names:
            hidden_output = self.tag_fc[tag_name](sentence_repr)
            tag_scores = F.log_softmax(hidden_output, dim=1)
            
            for word_idx, word_scores in enumerate(tag_scores):
                all_word_scores[word_idx].append(word_scores)

        # Shape: (sentence_len, 9, num_tag_output)
        return all_word_scores

    def training_step(self, sentence, batch_idx):
        """Predicts the output of the provided input for the model and calculates loss over it"""
        self.init_word_hidden()
        outputs = self.forward(sentence)
        # Shape: (sentence_len, 9, num_tag_output)

        loss = self.nll_loss(sentence, outputs)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, sentence, batch_idx):
        """Handles one single validation step, computes its loss and updates accuracy"""
        self.init_word_hidden()
        outputs = self.forward(sentence)
        # Shape: (sentence_len, 9, num_tag_output)

        loss = self.nll_loss(sentence, outputs)
        avg_acc, acc_by_tag = self.accuracy(sentence, outputs)

        return {'val_loss': loss, 'val_acc': avg_acc, 'acc_by_tag': acc_by_tag}

    def validation_epoch_end(self, outputs):
        """Called when an validation epoch ends, this prints out the average loss and accuracy"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        print('Validation loss is %.2f, validation accuracy is %.2f%%' % (avg_loss, avg_acc * 100))

        log = {'val_loss': avg_loss, 'val_acc': avg_acc}

        for idx, tag_name in enumerate(self.vocab.tag_names):
            avg_tag_acc = torch.stack([x['acc_by_tag'][idx] for x in outputs]).mean()
            log['tag_acc_%s' % tag_name] = avg_tag_acc

        return {'avg_val_loss': avg_loss, 'log': log}

    def nll_loss(self, sentence, outputs):
        """Calculates NLL loss over the combination of the predicted output and the ground truth"""
        loss_all_words = 0.0
        for word_idx in range(len(sentence)):
            output = outputs[word_idx]
            target = sentence[word_idx][2]

            try:
                loss_per_tag = [F.nll_loss(output[tag_idx].unsqueeze(0), target[tag_idx]) for tag_idx in range(self.tag_len)]
                loss_all_words += sum(loss_per_tag)
            except Exception as e:
                print(e)
                print('output 5: %s' % str(output[5].unsqueeze(0)))
                print('target 5: %s' % target[5])

        return loss_all_words / len(sentence)

    def accuracy(self, sentence, outputs):
        """Calculates the summed/mean accuracy for this sentence as well as the accuracy by tag"""
        sum_accuracy = 0.0
        sentence_len = len(sentence)
        sum_acc_by_tag = [0 for i in range(self.tag_len)]
        for word_idx in range(sentence_len):
            output = outputs[word_idx]
            target = sentence[word_idx][2]

            try:
                acc = [(torch.argmax(output[i]) == target[i]).float() for i in range(self.tag_len)] # Accuracy per tag per word
                sum_accuracy += sum(acc) / self.tag_len # Accuracy per word
                sum_acc_by_tag = add_element_wise(sum_acc_by_tag, acc)

            # During development this happened once or twice, should not happen anymore, but let's leave it in there
            except Exception as e:
                print(e)

        acc_by_tag = [acc / sentence_len for acc in sum_acc_by_tag]

        return sum_accuracy / sentence_len, acc_by_tag

    def configure_optimizers(self):
        """Returns the correctly configured optimizers"""
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        """A shortcut to the dataloader used for training data"""
        return DataLoader(self.train_data, batch_size=1, shuffle=True, num_workers=cpu_count())

    def val_dataloader(self):
        """A link to a dataloader that provides training data"""
        return DataLoader(self.val_data, batch_size=1, num_workers=cpu_count())
