import os
import pickle
from dataclasses import dataclass
from typing import Dict, List

import pyconll
import torch.utils.data


@dataclass
class Vocab:
    words: Dict[str, int]
    chars: Dict[str, int]
    tags: List[Dict[str, int]]
    inverted_tags: List[Dict[int, str]] = None

class PerseusDataset(torch.utils.data.Dataset):
    """This holds all the convenience methods for a dataset, such as loading as well as 
    implementing methods to make it usable with a dataloader"""

    dataset_fn = None
    vocab_fn = "vocab.p"
    number_of_tag_categories = 9

    def __init__(self, data_dir):
        """"Initializes the dataset from the provided input data url"""
        assert self.dataset_fn is not None, "You need to use the subclasses of PerseusDataset"

        self.sentences = []
        init_vocab = False
        vocab_path = os.path.join(data_dir, self.vocab_fn)
        if os.path.isfile(vocab_path):
            print("Loading vocabulary from cache: %s" % vocab_path)
            with open(vocab_path, "rb") as f:
                self.vocab = pickle.load(f)
        else:
            print("Creating vocabulary")
            init_vocab = True
            self.vocab = Vocab(
                words={"<UNK>": 0},
                chars={"<UNK>": 0},
                tags=[{"<UNK>": 0} for _ in range(self.number_of_tag_categories)]
            )

        input_body = pyconll.load_from_file(os.path.join(data_dir, self.dataset_fn))
        for sentence in input_body:
            tokenized_sentence = []
            for token in sentence:
                word = token.form
                characters = list(word)
                tags = list(token.xpos)
                assert len(tags) == 9
                tokenized_sentence.append(self.get_ids_and_create_vocab(word, characters, tags) if init_vocab else PerseusDataset.get_ids(self.vocab, word, characters, tags))

            self.sentences.append(tokenized_sentence)

        if init_vocab:
            self.vocab.inverted_tags = [{v: k for k, v in tag.items()} for tag in self.vocab.tags]

            with open(vocab_path, "wb") as vocab_file:
                pickle.dump(self.vocab, vocab_file, protocol=pickle.HIGHEST_PROTOCOL)
                print("Saved vocabulary to cache: %s" % vocab_path)


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

    def get_ids_and_create_vocab(self, word, characters, tags):
        if word not in self.vocab.words: self.vocab.words[word] = len(self.vocab.words)
        word_id = self.vocab.words[word]

        character_ids = []
        for character in characters:
            if character not in self.vocab.chars: self.vocab.chars[character] = len(self.vocab.chars)
            character_ids.append(self.vocab.chars[character])

        tag_ids = []
        for idx, tag in enumerate(tags):
            if tag not in self.vocab.tags[idx]: self.vocab.tags[idx][tag] = len(self.vocab.tags[idx])
            tag_ids.append(self.vocab.tags[idx][tag])

        return word_id, character_ids, tag_ids

    @staticmethod
    def get_ids(vocab, word, characters, tags):
        if word in vocab.words: word_id = vocab.words[word]
        else: word_id = vocab.words["<UNK>"]

        character_ids = []
        for character in characters:
            if character in vocab.chars: character_ids.append(vocab.chars[character])
            else: character_ids.append(vocab.chars["<UNK>"])

        tag_ids = []
        for idx, tag in enumerate(tags):
            if tag in vocab.tags[idx]: tag_ids.append(vocab.tags[idx][tag])
            else: tag_ids.append(vocab.tags[idx]["<UNK>"])

        return word_id, character_ids, tag_ids

class PerseusTrainingDataset(PerseusDataset):
    dataset_fn = "grc_perseus-ud-train.conllu"

class PerseusValidationDataset(PerseusDataset):
    dataset_fn = "grc_perseus-ud-dev.conllu"
