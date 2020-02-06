import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

class LSTMCharTagger(nn.Module):
    def __init__(self, embedding_dim_words, embedding_dim_chars, hidden_dim_words, hidden_dim_chars, vocab_size, tagset_size, charset_size):
        """LSTM Part-of-Speech Tagger Augmented with Character level features
        
        Atttributes:
            embedding_dim_words: Embedding dimension of word features to input to LSTM word level
            embedding_dim_chars: Embedding dimension of word features to input to character level
            hidden_dim_words: Output size of the LSTM word level
            hidden_dim_chars: Output size of the LSTM character level
            vocab_size: Size of the vocabulary of characters
            tagset_size: Size of the set of labels
            charset_size: Size of the vocabulary of characters
        """
        super(LSTMCharTagger, self).__init__()
        self.hidden_dim_words = hidden_dim_words
        self.hidden_dim_chars = hidden_dim_chars
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim_words)
        self.char_embeddings = nn.Embedding(charset_size, embedding_dim_chars)
        self.lstm_char = nn.LSTM(embedding_dim_chars, hidden_dim_chars)
        self.lstm_words = nn.LSTM(embedding_dim_words + hidden_dim_chars, hidden_dim_words)
        self.hidden2tag = nn.Linear(hidden_dim_words, tagset_size)
        self.hidden_char = self.init_hidden(c=False)
        self.hidden_words = self.init_hidden(c=True)
    
    def init_hidden(self, c=True):
        """Initialize hidden state of LSTMs
        
        Args:
            c(boolean): return initialized hidden state for LSTM word level if true
        
        """
        if c:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim_words)),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_dim_words)))
        else:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim_chars)),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_dim_chars)))
    
    
    def forward(self, sentence_seq, words_tensor_dict):
        """Forward propagation
        
        Args:
            sentence_seq(list): Sequence of indexis related to the corresponding sentence words
            words_tensor_dict(dict): Dictionary of tensors of words at the character level
        
        Returns:
            tensor: Labels predicted (POS) for the sequence
        """
        # embeds = self.word_embeddings(sentence)
        for ix, word_idx in enumerate(sentence_seq):
            
            # Char level
            word_chars_tensors = words_tensor_dict[int(word_idx)]
            char_embeds = self.char_embeddings(word_chars_tensors)
            
            # Remember that the input of LSTM is a 3D Tensor:
            # The first axis is the sequence itself, 
            # the second indexes instances in the mini-batch, and 
            # the third indexes elements of the input.
            lstm_char_out, self.hidden_char = self.lstm_char(
                char_embeds.view(len(char_embeds), 1, -1), self.hidden_char)
            
            # Word level
            embeds = self.word_embeddings(word_idx)
            # Now here we will only keep the final hidden state of the character level LSTM
            # i.e lstm_char_out[-1]
            embeds_cat = torch.cat((embeds, lstm_char_out[-1]), dim=1)
            
            lstm_out, self.hidden_words = self.lstm_words(embeds_cat, self.hidden_words)
            tag_space = self.hidden2tag(lstm_out.view(1, -1))
            
            tag_score = F.log_softmax(tag_space, dim=1)
            if ix==0:
                tag_scores = tag_score
            else:
                tag_scores = torch.cat((tag_scores, tag_score), 0)
        
        return tag_scores