import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class LSTMCharTagger(pl.LightningModule):
    """LSTM part-os-speech (PoS) tagger."""

    def __init__(self, hparams, train_data, val_data):
        super(LSTMCharTagger, self).__init__()
        self.hparams = hparams
        self.train_data = train_data
        self.val_data = val_data

        self.directions = 1
        self.num_layers = 1

        self.word_embeddings = nn.Embedding(len(self.train_data.word_ids), self.hparams.word_embedding_dim)

        self.word_lstm = nn.LSTM(
            self.hparams.word_embedding_dim,
            self.hparams.word_lstm_hidden_dim,
            bidirectional=self.directions > 1,
            dropout=0,
            num_layers=self.num_layers
        )

        self.first_tag_only = False

        tag_fc = []
        for idx in range(1 if self.first_tag_only else len(self.train_data.tag_ids)):
            num_tag_outputs = len(self.train_data.tag_ids[idx])
            fc = nn.Linear(self.hparams.word_lstm_hidden_dim, num_tag_outputs)
            tag_fc.append(fc)

        self.tag_fc = nn.ModuleList(tag_fc)

        self.init_word_hidden()

    def init_word_hidden(self):
        """Initialise word LSTM hidden state."""

        # TODO: shouldn't we initialize this differently?
        self.word_lstm_hidden = (
            torch.zeros(self.directions * self.num_layers, 1, self.hparams.word_lstm_hidden_dim),
            torch.zeros(self.directions * self.num_layers, 1, self.hparams.word_lstm_hidden_dim),
        )

    def forward(self, sentence):
        x = torch.tensor([word for word, _, _ in sentence])
        # Shape: (sentence_len, )

        x = self.word_embeddings(x)
        # Shape: (sentence_len, word_embedding_dim)

        x = x.view(len(sentence), self.hparams.batch_size, self.hparams.word_embedding_dim)
        # Shape: (sentence_len, batch_size, word_embedding_dim)

        x, self.word_lstm_hidden = self.word_lstm(x , self.word_lstm_hidden)
        # Shape: (sentence_len, batch_size, word_lstm_hidden_dim)

        all_word_scores = [[] for _ in range(len(sentence))]
        for idx in range(len(self.tag_fc)):
            hidden_output = self.tag_fc[idx](x)
            # Shape: (sentence_len, num_tag_output)

            tag_scores = F.log_softmax(hidden_output).squeeze(1)
            # Shape: (sentence_len, num_tag_output)

            for word_idx, word_score in enumerate(tag_scores):
                all_word_scores[word_idx].append(word_score)

        # Shape: (sentence_len, 9, num_tag_output)
        return all_word_scores

    def nll_loss(self, sentence, outputs):
        loss_all_sentences = 0.0
        for word_idx in range(len(sentence)):
            output = outputs[word_idx]
            target = sentence[word_idx][2]

            losses = [F.nll_loss(output[i].unsqueeze(0), target[i]) for i in range(1 if self.first_tag_only else 9)]
            loss_all_sentences += sum(losses)
        
        return loss_all_sentences / len(sentence)
    
    def training_step(self, sentence, batch_idx):
        outputs = self.forward(sentence)
        # Shape: (sentence_len, 9, num_tag_output)

        loss = self.nll_loss(sentence, outputs)

        self.init_word_hidden()

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, sentence, batch_idx):
        outputs = self.forward(sentence)
        # Shape: (sentence_len, 9, num_tag_output)

        loss = self.nll_loss(sentence, outputs)
    
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print('Validation loss is %.2f' % avg_loss)

        log = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': log}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=1, shuffle=True, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, num_workers=1)
