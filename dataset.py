import pyconll
import torch.utils.data
from gensim.models import KeyedVectors


class PerseusDataset(torch.utils.data.Dataset):
    """This holds all the convenience methods for a dataset, such as loading as well as 
    implementing methods to make it usable with a dataloader"""

    def __init__(self, url, embeddings, tag_ids=None):
        """"Initializes the dataset from the provided input data url"""
        self.url = url
        input_body = pyconll.load_from_file(url)
        self.number_of_tag_categories = 9
        self.embeddings = embeddings
        self.tag_ids = [{"<UNK>": 0} for _ in range(self.number_of_tag_categories)] if not tag_ids else tag_ids

        self.sentences = []
        for sentence in input_body:
            tokenized_sentence = []
            for token in sentence:
                word = token.form
                characters = list(word)
                tags = list(token.xpos)
                assert len(tags) == 9
                tokenized_sentence.append(self.get_embeddings(word, characters, tags))

            self.sentences.append(tokenized_sentence)

        """
            self.sentences = [
                [(word, char_id[], tags[]), (word, char[], tags[]), ...],
                [(word, char[], tags[]), (word, char[], tags[]), ...],
                ...
            ]
        """

    def __len__(self):
        """Returns the length of the amount of sentences in this dataset"""
        return len(self.sentences)

    def __getitem__(self, index):
        """Gets the item at the provided index in this dataset"""
        return self.sentences[index]

    def get_embeddings(self, word, characters, tags):
        if word in self.embeddings.word: word_id = torch.FloatTensor(self.embeddings.word[word])
        else: word_id = self.embeddings.unknown_word()

        character_ids = []
        for character in characters:
            if character in self.embeddings.char: character_ids.append(torch.FloatTensor(self.embeddings.char[character]))
            else: character_ids.append(self.embeddings.unknown_char())

        tag_ids = []
        for idx, tag in enumerate(tags):
            if tag not in self.tag_ids[idx]: self.tag_ids[idx][tag] = len(self.tag_ids[idx])
            tag_ids.append(self.tag_ids[idx][tag])

        return word_id, character_ids, tag_ids

class Embeddings():
    def __init__(self, data_dir):
        self.word = KeyedVectors.load_word2vec_format(data_dir + "/word_embeddings.bin")
        self.word_dim = self.word.vector_size
        self.char = KeyedVectors.load_word2vec_format(data_dir + "/char_embeddings.bin")
        self.char_dim = self.char.vector_size

    def unknown_word(self):
        return torch.rand(self.word_dim)

    def unknown_char(self):
        return torch.rand(self.char_dim)
