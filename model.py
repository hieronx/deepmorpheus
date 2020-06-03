import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from util import debug_mode

TAG_ID_TO_NAME = ["word_type", "person", "number", "tense", "mode", "voice", "gender", "case", "degree_of_comparison"]

def add_element_wise(list1, list2):
    return [a + b for a, b in zip(list1, list2)]


class LSTMCharTagger(pl.LightningModule):
    """LSTM part-os-speech (PoS) tagger."""

    def __init__(self, hparams, train_data, val_data):
        super(LSTMCharTagger, self).__init__()
        self.hparams = hparams
        self.train_data = train_data
        self.val_data = val_data

        self.single_output = True
        self.directions = 2
        self.num_layers = 2
        self.bs = 1

        self.word_embeddings = nn.Embedding(len(self.train_data.word_ids), self.hparams.word_embedding_dim)
        self.char_embeddings = nn.Embedding(len(self.train_data.character_ids), self.hparams.char_embedding_dim)

        self.word_lstm = nn.LSTM(
            self.hparams.word_embedding_dim + self.hparams.char_lstm_hidden_dim * self.directions,
            self.hparams.word_lstm_hidden_dim,
            bidirectional=self.directions > 1,
            dropout=0.2,
            num_layers=self.num_layers
        )

        self.char_lstm = nn.LSTM(
            self.hparams.char_embedding_dim,
            self.hparams.char_lstm_hidden_dim,
            bidirectional=self.directions > 1,
            dropout=0.2,
            num_layers=self.num_layers
        )

        self.enable_char_level = True

        self.cls_fc_dim = 512
        self.cls_fc = nn.Linear(self.hparams.word_lstm_hidden_dim * self.directions, self.cls_fc_dim)
        self.dropout = nn.Dropout(0.2)

        tag_fc = []
        for idx in range(len(self.train_data.tag_ids) if not self.single_output else 1):
            num_tag_outputs = len(self.train_data.tag_ids[idx])
            fc = nn.Linear(self.cls_fc_dim, num_tag_outputs)
            tag_fc.append(fc)

        self.tag_fc = nn.ModuleList(tag_fc)

        self.init_word_hidden()
        self.init_char_hidden()

    def init_word_hidden(self):
        """Initialise word LSTM hidden state."""

        # TODO: shouldn't we initialize this differently?
        self.word_lstm_hidden = (
            torch.zeros(self.directions * self.num_layers, 1, self.hparams.word_lstm_hidden_dim).to(self.device),
            torch.zeros(self.directions * self.num_layers, 1, self.hparams.word_lstm_hidden_dim).to(self.device),
        )

    def init_char_hidden(self):
        """Initialise char LSTM hidden state."""
        self.char_lstm_hidden = (
            torch.zeros(self.directions * self.num_layers, 1, self.hparams.char_lstm_hidden_dim).to(self.device),
            torch.zeros(self.directions * self.num_layers, 1, self.hparams.char_lstm_hidden_dim).to(self.device),
        )

    def forward(self, sentence):
        words = torch.tensor([word for word, _, _ in sentence]).to(self.device)
        # Shape: (sentence_len, )

        word_embeddings = self.word_embeddings(words)
        # Shape: (sentence_len, word_embedding_dim)

        word_embeddings_bs = word_embeddings.view(len(sentence), self.hparams.batch_size, self.hparams.word_embedding_dim)
        # Shape: (sentence_len, batch_size, word_embedding_dim)

        chars_reprs = []
        if self.enable_char_level:
            for word_idx in range(len(sentence)):
                self.init_char_hidden()
                chars = torch.tensor(sentence[word_idx][1]).to(self.device)

                chars_repr = None  # Character-level representation.
                # Character-level representation is the LSTM output of the last character.
                for char in chars:
                    char_embed = self.char_embeddings(char)
                    chars_repr, self.char_lstm_hidden = self.char_lstm(
                        char_embed.view(1, self.bs, self.hparams.char_embedding_dim), self.char_lstm_hidden
                    )
                
                chars_repr = chars_repr.view(1, self.hparams.char_lstm_hidden_dim * self.directions)
                chars_reprs.append(chars_repr)
        
        chars_reprs = torch.stack(chars_reprs).squeeze(1)
        word_repr = torch.cat([chars_reprs, word_embeddings], dim=1)
        
        sentence_repr, self.word_lstm_hidden = self.word_lstm(
            # Each sentence embedding dimensions are word embedding dimensions + character representation dimensions
            word_repr.view(len(sentence), self.bs, self.hparams.word_embedding_dim + self.hparams.char_lstm_hidden_dim * self.directions),
            self.word_lstm_hidden,
        )
        # Shape: (sentence_len, batch_size, word_lstm_hidden_dim)

        hidden_repr = self.cls_fc(sentence_repr)
        regularized_repr = self.dropout(hidden_repr)

        all_word_scores = [[] for _ in range(len(sentence))]
        for idx in range(len(self.tag_fc)):
            hidden_output = self.tag_fc[idx](regularized_repr)
            # Shape: (sentence_len, 1, num_tag_output)

            tag_scores = F.log_softmax(hidden_output, dim=2).squeeze(1)
            # Shape: (sentence_len, num_tag_output)

            for word_idx, word_score in enumerate(tag_scores):
                all_word_scores[word_idx].append(word_score)

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


    # Nu: avg nll loss per woord
    # nll_loss([output_word1, output_word2, ...], [target_word1, target_word2, ...])

    def nll_loss(self, sentence, outputs):
        loss_all_words = 0.0
        for word_idx in range(len(sentence)):
            output = outputs[word_idx]
            target = sentence[word_idx][2]

            try:
                loss_per_tag = [F.nll_loss(output[i].unsqueeze(0), target[i]) for i in range(len(self.tag_fc))]
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
