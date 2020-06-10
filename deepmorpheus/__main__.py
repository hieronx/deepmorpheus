import os
import pickle
from argparse import ArgumentParser, Namespace

import torch

from deepmorpheus.dataset import PerseusDataset
from deepmorpheus.model import LSTMCharTagger

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

def attempt_checkpoint_load(checkpoint_path, force_compatability=False):
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

    # Creates the model and loads the state dict, then return it
    model = LSTMCharTagger(hparams, vocab)
    model.load_state_dict(checkpoint['state_dict'])
    return model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="data")
    parser.add_argument('--ckpt-name', type=str, default="inference.ckpt")
    parser.add_argument('--input-file', type=str, default="test_input.txt")
    args = parser.parse_args()

    # Try to load vocab.p or abort
    vocab_path = os.path.join(args.data_dir, PerseusDataset.vocab_fn)
    vocab = attempt_vocab_load(vocab_path)

    # Try to load input file as list of lines, or abort
    input_file = attempt_input_load(args.input_file)

    # List of sentences ready to infer on, and list of words by sentence, used to translate back from index to word
    sentences = []
    words_per_sentence = []

    # For each of the lines in the input, make a separate sentence and split into list of words
    for line in input_file:
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
    model = attempt_checkpoint_load(os.path.join(args.data_dir, args.ckpt_name))
    
    for sentence_idx, sentence in enumerate(sentences):
        model.init_word_hidden()
        output = model(sentence)
        print()

        for word_idx, word_output in enumerate(output):
            tags = []
            for tag_idx, tag_output in enumerate(word_output):
                tag_output_id = torch.argmax(tag_output).item()
                tags.append(vocab.inverted_tags[tag_idx][tag_output_id])
            
            print('%s\t\t%s' % (words_per_sentence[sentence_idx][word_idx], "".join(tags)))
