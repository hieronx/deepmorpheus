import os
import pickle
from argparse import ArgumentParser, Namespace

import torch

from dataset import PerseusDataset
from model import LSTMCharTagger

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="data")
    parser.add_argument('--ckpt-name', type=str, default="inference.ckpt")
    parser.add_argument('--input-file', type=str, default="test_input.txt")
    args = parser.parse_args()

    vocab_path = os.path.join(args.data_dir, PerseusDataset.vocab_fn)
    if not os.path.isfile(vocab_path):
        print("Vocabulary needs to be located here: %s" % vocab_path)
        exit()

    print("Loading vocabulary from cache: %s" % vocab_path)
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    if not os.path.isfile(args.input_file):
        print("Input file does not exist: %s" % args.input_file)
        exit()

    sentences = []
    words_per_sentence = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            words = [word.strip() for word in line.split(" ")]
            sentence = []
            for word in words:
                chars = list(word)
                sentence.append(PerseusDataset.get_ids(vocab, word, chars, []))
            sentences.append(sentence)
            words_per_sentence.append(words)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device: %s" % device)
    
    ckpt_path = os.path.join(args.data_dir, args.ckpt_name)
    print("Loading model from checkpoint: %s" % ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
    ckpt_hparams = checkpoint.get('hparams')
    hparams = Namespace(**ckpt_hparams)

    # TODO: remove this when we create a new checkpoint
    hparams.disable_bidirectional = False
    hparams.disable_char_level = False
    hparams.dropout = 0.3

    model = LSTMCharTagger(hparams, vocab)
    model.load_state_dict(checkpoint['state_dict'])

    for sentence_idx, sentence in enumerate(sentences):
        model.init_word_hidden()
        output = model(sentence)
        for word_idx, word_output in enumerate(output):
            tags = []
            for tag_idx, tag_output in enumerate(word_output):
                tag_output_id = torch.argmax(tag_output).item()
                tags.append(vocab.inverted_tags[tag_idx][tag_output_id])
            
            print('%s\t\t%s' % (words_per_sentence[sentence_idx][word_idx], "".join(tags)))
        print()
