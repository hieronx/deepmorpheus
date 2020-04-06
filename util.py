import torch


def parse_into_list(body):
    """Turns the body of text into a list of tuples of the word and their respective tag"""
    data = []
    for sentence in body:
        sentence_words = []
        sentence_tags = []
        for token in sentence:
            sentence_words.append(token.form)
            sentence_tags.append(token.upos)

        if len(sentence_words) > 0:
            data.append((sentence_words, sentence_tags))

    return data


def make_ixs(seq, to_ix, device):
    ixs = torch.tensor([to_ix[w] if w in to_ix else to_ix["<UNK>"] for w in seq]).to(
        device
    )
    return ixs
