import pickle
from dataclasses import dataclass, field
from typing import Dict, List
from os import path

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
    tag_names: List[str] = field(default_factory=lambda: ["word_type", "person", "number", "tense", "mode", "voice", "gender", "case", "degree_of_comparison"]) # This is just a lookup table to make it easier for us puny humans to understand what the tagger is tagging

class PerseusDataset(torch.utils.data.Dataset):
    """This holds all the convenience methods for a dataset, such as loading as well as 
    implementing methods to make it usable with a dataloader
    
    The dataformat we use in this dataloader is as follows: 
        self.sentences = [
            [(word, char[], tags[]), (word, char[], tags[]), ...],
            [(word, char[], tags[]), (word, char[], tags[]), ...],
            ...
        ]
    """

    # Needs to be overriden by the subclass to point to the data file
    dataset_fn = None
    # The number of tag categories, this SHOULD never change
    NUM_TAGS = 9

    def __init__(self, data_dir, language):
        """"Initializes the dataset from the provided input data url"""
        assert self.dataset_fn is not None, "You need to use the subclasses of PerseusDataset"

        # First we try to load the vocab pickle, if it doesn't exist yet we must create it
        vocab_path = path.join(data_dir, "vocab-%s.p" % language)
        init_vocab = not path.isfile(vocab_path)
        if not init_vocab:
            print("Loading vocabulary from cache: %s" % vocab_path)
            with open(vocab_path, "rb") as f:
                self.vocab = pickle.load(f)
        else:
            print("Creating vocabulary for %s" % language)
            self.vocab = Vocab(
                words={"<UNK>": 0},
                chars={"<UNK>": 0},
                tags=[{"<UNK>": 0} for _ in range(self.NUM_TAGS)]
            )

        # Now that we've decided if we have a premade vocab we're parsing the dataset into tokenized data that the model can work with
        self.sentences = []
        for sentence in pyconll.load_from_file(path.join(data_dir, self.dataset_fn)):
            tokenized_sentence = []
            for token in sentence:
                word, characters, tags = token.form, list(token.form), token.xpos
                assert len(tags) == 9, "Tags should always have a length of 9"

                tokenized_sentence.append(PerseusDataset.get_ids(self.vocab, word, characters, tags, expand_vocab=init_vocab))

            self.sentences.append(tokenized_sentence)

        # If we have just created the vocabulary, let's save it just in case
        if init_vocab: self.save_vocab(vocab_path)

    def save_vocab(self, vocab_path):
        """This function saves the vocabulary file to the disk location provided"""
        self.vocab.inverted_tags = [{v: k for k, v in tag.items()} for tag in self.vocab.tags]

        with open(vocab_path, "wb") as vocab_file:
            pickle.dump(self.vocab, vocab_file, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved vocabulary to cache: %s" % vocab_path)


    def __len__(self):
        """Returns the number of sentences in this dataset"""
        return len(self.sentences)

    def __getitem__(self, index):
        """Gets the item at the provided index in this dataset"""
        return self.sentences[index]

    @staticmethod
    def get_ids(vocab, word, characters, tags, expand_vocab = False):
        """This function assigns ids to words, characters and tags if we are expanding the provided vocab,
        otherwise it just does a lookup in the provided vocab and returns an unknown token (0)"""

        # Check if the word already exists in the vocab
        if word not in vocab.words:
            # Check if we're can add unknown words, if not, return UKNOWN, otherwise add it
            if expand_vocab:
                vocab.words[word] = len(vocab.words)
                word_id = vocab.words[word]
            else:
                word_id = vocab.words["<UNK>"]
        else:
            word_id = vocab.words[word]

        character_ids = []
        # Check if the characters already exists in the vocab
        for character in characters:
            # Check if we're expanding, if not, return UNKOWN, otherwise add it and return its id
            if character not in vocab.chars: 
                if expand_vocab:
                    vocab.chars[character] = len(vocab.chars)
                    character_ids.append(vocab.chars[character])
                else:
                    character_ids.append(vocab.chars["<UNK>"])
            else:
                character_ids.append(vocab.chars[character])

        tag_ids = []
        # For each of the possible tags for this word
        for idx, tag in enumerate(tags):
            # Check if the tag is already in vocab
            if tag not in vocab.tags[idx]: 
                # If we're expanding, add it to vocab, otherwise return an UKNOWN
                if expand_vocab:
                    vocab.tags[idx][tag] = len(vocab.tags[idx])
                    tag_ids.append(vocab.tags[idx][tag])
                else:
                    tag_ids.append(vocab.tags[idx]["<UNK>"])
            else:
                tag_ids.append(vocab.tags[idx][tag])

        # Now we have one word (one data entry) ready for traing on the model
        return word_id, character_ids, tag_ids

class AncientGreekTrainDataset(PerseusDataset):
    dataset_fn = "grc_perseus-ud-train.conllu"

class AncientGreekValDataset(PerseusDataset):
    dataset_fn = "grc_perseus-ud-dev.conllu"

class LatinTrainDataset(PerseusDataset):
    dataset_fn = "la_perseus-ud-train.conllu"

class LatinTestDataset(PerseusDataset):
    dataset_fn = "la_perseus-ud-test.conllu"
