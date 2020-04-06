import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

# Source: https://github.com/sherif7810/lstm_pos_tagger/blob/master/main.py
class LSTMCharTagger(nn.Module):
    """LSTM part-os-speech (PoS) tagger."""

    def __init__(
        self,
        word_embedding_dim,
        char_embedding_dim,
        char_lstm_hidden_dim,
        word_lstm_hidden_dim,
        word_dict_size,
        char_dict_size,
        tagset_size,
        device,
    ):
        """
        word_embedding_dim: size of the word embedding vectors
        char_embedding_dim: size of the char embedding vectors
        char_lstm_hidden_dim: size of the Char-LSTM hidden layer 
        word_lstm_hidden_dim: size of the Word-LSTM hidden layer
        word_dict_size: # of total words
        char_size: # of total characters
        tagset_size: # of output labels
        device: CPU or GPU device
        """

        super(LSTMCharTagger, self).__init__()
        self.char_lstm_hidden_dim = char_lstm_hidden_dim
        self.word_lstm_hidden_dim = word_lstm_hidden_dim

        self.word_embeddings = nn.Embedding(word_dict_size, word_embedding_dim)
        self.char_embeddings = nn.Embedding(char_dict_size, char_embedding_dim)

        self.char_lstm = nn.LSTM(char_embedding_dim, char_lstm_hidden_dim)
        self.word_lstm = nn.LSTM(
            word_embedding_dim + char_lstm_hidden_dim, word_lstm_hidden_dim
        )

        self.hidden2tag = nn.Linear(word_lstm_hidden_dim, tagset_size)

        self.device = device

        self.init_char_hidden()
        self.init_word_hidden()

    def init_word_hidden(self):
        """Initialise word LSTM hidden state."""
        self.word_lstm_hidden = (
            torch.zeros(1, 1, self.word_lstm_hidden_dim).to(self.device),
            torch.zeros(1, 1, self.word_lstm_hidden_dim).to(self.device),
        )

    def init_char_hidden(self):
        """Initialise character LSTM hidden state."""
        self.char_lstm_hidden = (
            torch.zeros(1, 1, self.char_lstm_hidden_dim).to(self.device),
            torch.zeros(1, 1, self.char_lstm_hidden_dim).to(self.device),
        )

    def forward(self, word_characters_ixs):
        sentence_length = len(
            word_characters_ixs
        )  # this is only the # of unique words in the sentence, because of the way word_character_ixs is constructed

        word_repr = []
        for word_ix, char_ixs in word_characters_ixs.items():
            word_embed = self.word_embeddings(word_ix.to(self.device))

            # TODO: what happens if we remove this?
            self.init_char_hidden()  # Don't account for the characters of the previous word

            chars_repr = None  # Character-level representation.
            # Character-level representation is the LSTM output of the last character.
            for char_ix in char_ixs:
                char_embed = self.char_embeddings(char_ix)
                chars_repr, self.char_lstm_hidden = self.char_lstm(
                    char_embed.view(1, 1, -1), self.char_lstm_hidden
                )

            word_repr.append(word_embed)
            word_repr.append(chars_repr.view(1, -1))

        word_repr = torch.cat(word_repr, 1)  # From row to column

        sentence_repr, self.word_lstm_hidden = self.word_lstm(
            # Each sentence embedding dimensions are word embedding dimensions + character representation dimensions
            word_repr.view(sentence_length, 1, -1),
            self.word_lstm_hidden,
        )

        tag_space = self.hidden2tag(sentence_repr.view(sentence_length, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores
