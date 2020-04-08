import torch.utils.data
import pyconll

class PerseusCoNLLUDataset(torch.utils.data.Dataset):

    def __init__(self, url):
        """"Initializes the dataset from the provided input data url"""
        self.url = url
        input_body = pyconll.load_from_file(url)
        self.sentences = []
        for sentence in input_body:
            sentence_words = []
            sentence_tags = []
            for token in sentence:
                sentence_words.append(token.form)
                sentence_tags.append(token.upos)

            if len(sentence_words) > 0:
                self.sentences.append((sentence_words, sentence_tags))

    def __len__(self):
        """Returns the length of the amount of sentences in this dataset"""
        return len(self.sentences)

    def __getitem__(self, index):
        """Gets the item at the provided index in this dataset"""
        return self.sentences[index]

    def create_indices(self):
        """Creates the indeces that are used in the neural network, effectively turning text into numbers"""
        print('Creating indices for %s' % self.url)
        self.word_to_ix = {"<UNK>": 0}
        self.char_to_ix = {"<UNK>": 0}

        for words, _ in self.sentences:
            for word in words:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)

                for char in word:
                    if char not in self.char_to_ix:
                        self.char_to_ix[char] = len(self.char_to_ix)
        
        self.tag_to_ix = {}
        for _, tags in self.sentences:
            for tag in tags:
                if tag not in self.tag_to_ix:
                    self.tag_to_ix[tag] = len(self.tag_to_ix)
    
    def get_indices(self):
        """Returns a tuple of the word indices, char indices and tag indices"""
        if not 'word_to_ix' in self: self.create_indices()

        return (self.word_to_ix, self.char_to_ix, self.tag_to_ix)