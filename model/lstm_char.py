import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

class LSTMCharTagger(nn.Module):
    """LSTM part-os-speech (PoS) tagger."""

    def __init__(self, word_embedding_dim, char_embedding_dim, char_repr_dim, hidden_dim, vocab_size, chars_size, tagset_size, word_to_ix, char_to_ix, make_ixs):
        super(LSTMCharTagger, self).__init__()
        self.char_repr_dim = char_repr_dim
        self.hidden_dim = hidden_dim
        
        self.word_to_ix = word_to_ix
        self.char_to_ix = char_to_ix
        self.make_ixs = make_ixs

        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.char_embeddings = nn.Embedding(chars_size, char_embedding_dim)

        self.char_lstm = nn.LSTM(char_embedding_dim, char_repr_dim)
        self.word_lstm = nn.LSTM(word_embedding_dim + char_repr_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.char_lstm_hidden = (torch.zeros(1, 1, self.char_repr_dim),
                                 torch.zeros(1, 1, self.char_repr_dim))
        self.word_lstm_hidden = (torch.zeros(1, 1, self.hidden_dim),
                                 torch.zeros(1, 1, self.hidden_dim))

    def init_word_hidden(self):
        """Initialise word LSTM hidden state."""
        self.word_lstm_hidden = (torch.zeros(1, 1, self.hidden_dim),
                                 torch.zeros(1, 1, self.hidden_dim))

    def init_char_hidden(self):
        """Initialise character LSTM hidden state."""
        self.char_lstm_hidden = (torch.zeros(1, 1, self.char_repr_dim),
                                 torch.zeros(1, 1, self.char_repr_dim))

    def forward(self, sentence):
        sentence_length = len(sentence)
        word_characters_ixs = {}
        for word in sentence:
            word_ix = torch.tensor([self.word_to_ix[word]])
            char_ixs = self.make_ixs(word, self.char_to_ix)
            word_characters_ixs[word_ix] = char_ixs

        word_embeds = []
        for word_ix, char_ixs in word_characters_ixs.items():
            word_embed = self.word_embeddings(word_ix)

            self.init_char_hidden()
            c = None  # Character-level representation.
            # Character-level representation is the LSTM output of the last character.
            for char_ix in char_ixs:
                char_embed = self.char_embeddings(char_ix)
                c, self.char_lstm_hidden = self.char_lstm(
                    char_embed.view(1, 1, -1), self.char_lstm_hidden)
            word_embeds.append(word_embed)
            word_embeds.append(c.view(1, -1))
        word_embeds = torch.cat(word_embeds, 1)

        lstm_out, self.word_lstm_hidden = self.word_lstm(
            # Each sentence embedding dimensions are word embedding dimensions + character representation dimensions
            word_embeds.view(sentence_length, 1, -1),
            self.word_lstm_hidden)
        tag_space = self.hidden2tag(lstm_out.view(sentence_length, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
