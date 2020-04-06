import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.autograd as autograd
import pickle
import pyconll

from dataset import PerseusCoNLLUDataset
from util import make_ixs

model = pickle.load(open("data/model.p", "rb"))
device = "cpu"

train = PerseusCoNLLUDataset(pyconll.load_from_file(
    "data/perseus-conllu/grc_perseus-ud-train.conllu"
))
val = PerseusCoNLLUDataset(pyconll.load_from_file("data/perseus-conllu/grc_perseus-ud-dev.conllu"))
word_to_ix, char_to_ix, tag_to_ix = train.get_indices()

ix_to_tag = {v: k for k, v in tag_to_ix.items()}

sentence = val[0][0]
targets = val[0][1]

with torch.no_grad():
    word_characters_ixs = []
    for word in sentence:
        word_ix = torch.tensor([word_to_ix[word]]) if word in word_to_ix else torch.tensor([word_to_ix['<UNK>']])
        char_ixs = make_ixs(word, char_to_ix, device)
        word_characters_ixs.append((word_ix, char_ixs))

    inputs = make_ixs(sentence, word_to_ix, device)
    token_scores = model(word_characters_ixs)
    scores = [score.tolist() for score in token_scores]
    tag_ix = [score.index(max(score)) for score in scores]
    tags = [ix_to_tag[tag] if tag in ix_to_tag else '' for tag in tag_ix]

    for i, (word, tag) in enumerate(zip(sentence, tags)):
        print('%s = %s (should be %s)' % (word, tag, targets[i]))
