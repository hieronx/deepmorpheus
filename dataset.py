import pyconll
import torch.utils.data


class PerseusDataset(torch.utils.data.Dataset):
    """This holds all the convenience methods for a dataset, such as loading as well as 
    implementing methods to make it usable with a dataloader"""

    def __init__(self, url):
        """"Initializes the dataset from the provided input data url"""
        self.url = url
        input_body = pyconll.load_from_file(url)
        self.number_of_tag_categories = 9

        self.word_ids = {}
        self.character_ids = {}
        self.tag_ids = [{} for _ in range(self.number_of_tag_categories)]

        self.sentences = []
        for sentence in input_body:
            tokenized_sentence = []
            for token in sentence:
                word = token.form
                characters = list(word)
                tags = list(token.xpos)
                assert len(tags) == 9
                tokenized_sentence.append(self.get_ids(word, characters, tags))

            self.sentences.append(tokenized_sentence)

        """
            self.sentences = [
                [(word, char[], tags[]), (word, char[], tags[]), ...],
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

    def get_ids(self, word, characters, tags):
        if word not in self.word_ids: self.word_ids[word] = len(self.word_ids)
        word_id = self.word_ids[word]

        character_ids = []
        for character in characters:
            if character not in self.character_ids: self.character_ids[character] = len(self.character_ids)
            character_ids.append(self.character_ids[character])

        tag_ids = []
        for idx, tag in enumerate(tags):
            if tag not in self.tag_ids[idx]: self.tag_ids[idx][tag] = len(self.tag_ids[idx])
            tag_ids.append(self.tag_ids[idx][tag])

        return word_id, character_ids, tag_ids
