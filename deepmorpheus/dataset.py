import os
import pickle
from dataclasses import dataclass
from typing import Dict, List

import pyconll
import torch.utils.data


@dataclass
class Vocab:
    """This data holder class will hold the tokenized training data, this is necessary
    to tokenize input during inference"""
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
                assert len(tags) == 9, "Tags should always have a length of 9"
                tokenized_sentence.append(PerseusDataset.get_ids(self.vocab, word, characters, tags, expand_vocab=init_vocab))

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

    @staticmethod
    def get_ids(vocab, word, characters, tags, expand_vocab = False):
        if word not in vocab.words:
            if expand_vocab:
                vocab.words[word] = len(vocab.words)
                word_id = vocab.words[word]
            else:
                word_id = vocab.words["<UNK>"]
        else:
            word_id = vocab.words[word]

        character_ids = []
        for character in characters:
            if character not in vocab.chars: 
                if expand_vocab:
                    vocab.chars[character] = len(vocab.chars)
                    character_ids.append(vocab.chars[character])
                else:
                    character_ids.append(vocab.chars["<UNK>"])
            else:
                character_ids.append(vocab.chars[character])

        tag_ids = []
        for idx, tag in enumerate(tags):
            if tag not in vocab.tags[idx]: 
                if expand_vocab:
                    vocab.tags[idx][tag] = len(vocab.tags[idx])
                    tag_ids.append(vocab.tags[idx][tag])
                else:
                    tag_ids.append(vocab.tags[idx]["<UNK>"])
            else:
                tag_ids.append(vocab.tags[idx][tag])

        return word_id, character_ids, tag_ids

class PerseusTrainingDataset(PerseusDataset):
    dataset_fn = "grc_perseus-ud-train.conllu"

class PerseusValidationDataset(PerseusDataset):
    dataset_fn = "grc_perseus-ud-dev.conllu"
