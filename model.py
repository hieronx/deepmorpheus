import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from util import add_element_wise

TAG_ID_TO_NAME = ["word_type", "person", "number", "tense", "mode", "voice", "gender", "case", "degree_of_comparison"]


class LSTMCharTagger(pl.LightningModule):
    """LSTM part-os-speech (PoS) tagger."""

    def __init__(self, hparams, train_data, val_data, word_embedding_dim, char_embedding_dim):
        super(LSTMCharTagger, self).__init__()
        self.hparams = hparams
        self.train_data = train_data
        self.val_data = val_data
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim

        self.single_output = False
        self.directions = 1 if self.hparams.disable_bidirectional else 2
        self.hparams.num_lstm_layers = 2

        self.word_lstm_input_dim = self.word_embedding_dim if hparams.disable_char_level else self.word_embedding_dim + self.hparams.char_lstm_hidden_dim * self.directions
        self.word_lstm = nn.LSTM(
            self.word_lstm_input_dim,
            self.hparams.word_lstm_hidden_dim,
            bidirectional=self.directions > 1,
            dropout=self.hparams.dropout,
            num_layers=self.hparams.num_lstm_layers
        )
        self.init_word_hidden()

        if not hparams.disable_char_level:        
            self.char_lstm = nn.LSTM(
                self.char_embedding_dim,
                self.hparams.char_lstm_hidden_dim,
                bidirectional=self.directions > 1,
                dropout=self.hparams.dropout,
                num_layers=self.hparams.num_lstm_layers
            )
            self.init_char_hidden()

        tag_fc = []
        for idx in range(len(self.train_data.tag_ids) if not self.single_output else 1):
            num_tag_outputs = len(self.train_data.tag_ids[idx])
            fc = nn.Linear(self.hparams.word_lstm_hidden_dim * self.directions, num_tag_outputs)
            tag_fc.append(fc)

        self.tag_fc = nn.ModuleList(tag_fc)

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
        words = torch.stack([word.squeeze(0) for word, _, _ in sentence]).to(self.device)
        words_bs = words.view(len(sentence), self.hparams.batch_size, self.word_embedding_dim)

        word_repr = []
        if not self.hparams.disable_char_level:
            for word_idx in range(len(sentence)):
                self.init_char_hidden() # Don't store representation between words
                chars = torch.stack(sentence[word_idx][1]).to(self.device)

                # Character-level representation is the LSTM output of the last character.
                chars_repr = None
                for char in chars:
                    chars_repr, self.char_lstm_hidden = self.char_lstm(
                        char.view(1, self.hparams.batch_size, self.char_embedding_dim), self.char_lstm_hidden
                    )

                chars_repr = chars_repr.view(1, self.hparams.char_lstm_hidden_dim * self.directions)

                word_repr.append(words[0].unsqueeze(0))
                word_repr.append(chars_repr)
        else:
            for word_idx in range(len(sentence)):
                word_repr.append(words[0].unsqueeze(0))

        # Each sentence embedding dimensions are word embedding dimensions + character representation dimensions
        word_repr = torch.cat(word_repr, dim=1) # From row to column

        sentence_repr, self.word_lstm_hidden = self.word_lstm(
            word_repr.view(len(sentence), self.hparams.batch_size, self.word_lstm_input_dim),
            self.word_lstm_hidden,
        )

        sentence_repr = sentence_repr.view(len(sentence), self.hparams.word_lstm_hidden_dim * self.directions)

        all_word_scores = [[] for _ in range(len(sentence))]

        for idx in range(len(self.tag_fc)):
            hidden_output = self.tag_fc[idx](sentence_repr)
            tag_scores = F.log_softmax(hidden_output, dim=1)
            
            for word_idx, word_scores in enumerate(tag_scores):
                all_word_scores[word_idx].append(word_scores)

        # Shape: (sentence_len, 9, num_tag_output)
        return all_word_scores

    def training_step(self, sentence, batch_idx):
        self.init_word_hidden()
        outputs = self.forward(sentence)
        # Shape: (sentence_len, 9, num_tag_output)

        loss = self.nll_loss(sentence, outputs)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, sentence, batch_idx):
        self.init_word_hidden()
        outputs = self.forward(sentence)
        # Shape: (sentence_len, 9, num_tag_output)

        loss = self.nll_loss(sentence, outputs)
        avg_acc, acc_by_tag = self.accuracy(sentence, outputs)

        return {'val_loss': loss, 'val_acc': avg_acc, 'acc_by_tag': acc_by_tag}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        print('Validation loss is %.2f, validation accuracy is %.2f%%' % (avg_loss, avg_acc * 100))

        log = {'val_loss': avg_loss, 'val_acc': avg_acc}

        for i in range(len(self.tag_fc)):
            avg_tag_acc = torch.stack([x['acc_by_tag'][i] for x in outputs]).mean()
            log['tag_acc_%s' % TAG_ID_TO_NAME[i]] = avg_tag_acc

        return {'avg_val_loss': avg_loss, 'log': log}

    def nll_loss(self, sentence, outputs):
        loss_all_words = 0.0
        for word_idx in range(len(sentence)):
            output = outputs[word_idx]
            target = sentence[word_idx][2]

            try:
                loss_per_tag = [F.nll_loss(output[tag_idx].unsqueeze(0), target[tag_idx]) for tag_idx in range(len(self.tag_fc))]
                loss_all_words += sum(loss_per_tag)
            except Exception as e:
                print(e)
                print('output 5: %s' % str(output[5].unsqueeze(0)))
                print('target 5: %s' % target[5])

        return loss_all_words / len(sentence)

    def accuracy(self, sentence, outputs):
        sum_accuracy = 0.0
        sum_acc_by_tag = [0 for i in range(len(self.tag_fc))]
        for word_idx in range(len(sentence)):
            output = outputs[word_idx]
            target = sentence[word_idx][2]

            try:
                acc = [(torch.argmax(output[i]) == target[i]).float() for i in range(len(self.tag_fc))] # Accuracy per tag per word
                sum_accuracy += sum(acc) / len(self.tag_fc) # Accuracy per word
                sum_acc_by_tag = add_element_wise(sum_acc_by_tag, acc)

            except Exception as e:
                print(e)
                print('output 5: %s' % str(output[5].unsqueeze(0)))
                print('target 5: %s' % target[5])

        acc_by_tag = [acc / len(sentence) for acc in sum_acc_by_tag]

        return sum_accuracy / len(sentence), acc_by_tag

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=1, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, num_workers=4)