import os
import pickle
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from typing import Dict, List

import torch

from deepmorpheus.dataset import PerseusDataset
from deepmorpheus.model import LSTMCharTagger
from deepmorpheus.util import readable_conversion_file, tag_to_readable


def attempt_vocab_load(vocab_path):
    """This function will try to load the vocab file from data/vocab.p.
    If it fails it will abort execution since we need a vocabulary to correctly
    tokenize the input data"""
    if not os.path.isfile(vocab_path):
        print("Vocabulary needs to be located here: %s" % vocab_path)
        exit()
    
    print("Loading vocabulary from cache: %s" % vocab_path)
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab

def attempt_input_load(input_path):
    """Attempts to load the file at the provided path and return it as an array
    of lines. If the file does not exist we will exit the program since nothing
    useful can be done."""
    if not os.path.isfile(input_path):
        print("Input file does not exist: %s" % input_path)
        exit()

    print("Loading input from file: %s" % input_path)
    with open(input_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
    return lines

def attempt_checkpoint_load(checkpoint_path, vocab, device, force_compatibility=False):
    """This function tries to load a pytorch checkpoint, if it fails it aborts the program"""
    if not os.path.isfile(checkpoint_path):
        print("Model checkpoint file does not exist: %s" % checkpoint_path)
        exit()

    print("Loading model from checkpoint: %s" % checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    checkpoint_params = checkpoint.get('hparams')
    hparams = Namespace(**checkpoint_params)

    # Only turn this on if we need to load an older model which had different hparams
    if force_compatability:
        hparams.disable_bidirectional = False
        hparams.disable_char_level = False
        hparams.dropout = 0.3
        # vocab.tag_names = ["word_type", "person", "number", "tense", "mode", "voice", "gender", "case", "degree_of_comparison"]

    # Creates the model and loads the state dict, then return it
    model = LSTMCharTagger(hparams, vocab)
    model.load_state_dict(checkpoint['state_dict'])
    return model


@dataclass
class WordWithTags:
    word: str
    tags: Dict[str, str]
    readable_tags: str

    def __str__(self):
        return "%s: %s" % (self.word, self.readable_tags)

def tag_from_file(input_path, language="ancient-greek", data_dir="data"):
    """Loads from a specified file, loads the file and then forwards to the tag_from_lines function """
    # Try to load input file as list of lines, or abort
    input_file = attempt_input_load(input_path)
    return tag_from_lines(input_file, language, data_dir)

def tag_from_lines(input_lines, language="ancient-greek", data_dir="data"):
    assert language == "ancient-greek" or language == "latin", "Language parameter for tag_from_file() needs to be one of: [ancient-greek, latin]"

    # Try to load vocab.p or abort
    vocab_path = os.path.join(data_dir, "vocab-%s.p" % language)
    vocab = attempt_vocab_load(vocab_path)

    # List of sentences ready to infer on, and list of words by sentence, used to translate back from index to word
    sentences = []
    words_per_sentence = []

    # For each of the lines in the input, make a separate sentence and split into list of words
    for line in input_lines:
        sentence = []
        words = [word.strip() for word in line.split(" ")]
        # For each word in the sentence, tokenize it and its chars, we don't know the tags, so leave empty
        for word in words:
            sentence.append(PerseusDataset.get_ids(vocab, word, list(word), []))

        # Append these entries to both the data for inference and the lookup data for when the results are done
        sentences.append(sentence)
        words_per_sentence.append(words)

    # Let's see what device we can run this on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device: %s" % device)
    
    # Let's see if we can load the checkpoint
    model = attempt_checkpoint_load(os.path.join(data_dir, language + ".ckpt"), vocab, device, True)

    conversion = readable_conversion_file(os.path.join(data_dir, "tagconversion_en.csv"))
    
    return_output = []
    for sentence_idx, sentence in enumerate(sentences):
        model.init_word_hidden()
        output = model(sentence)

        sentence_output = []
        for word_idx, word_output in enumerate(output):
            tags = {}
            tag_str = ""
            for tag_idx, tag_output in enumerate(word_output):
                tag_output_id = torch.argmax(tag_output).item()
                tags[vocab.tag_names[tag_idx]] = vocab.inverted_tags[tag_idx][tag_output_id]
                tag_str += vocab.inverted_tags[tag_idx][tag_output_id]

            word_with_tags = WordWithTags(words_per_sentence[sentence_idx][word_idx], tags, tag_to_readable(tag_str, conversion))
            sentence_output.append(word_with_tags)
        
        return_output.append(sentence_output)
    
    return return_output
