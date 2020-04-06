import torch.utils.data

class PerseusCoNLLUDataset(torch.utils.data.Dataset):

    def __init__(self, input_body):
        self.sentences = []
        for sentence in input_body:
            sentence_words = []
            sentence_tags = []
            for token in sentence:
                sentence_words.append(token.form)
                sentence_tags.append(token.upos)

            if len(sentence_words) > 0:
                self.sentences.append((sentence_words, sentence_tags))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

    def create_indices(self):
        print('Creating indices...')
        self.word_to_ix = {"<UNK>": 0}
        self.char_to_ix = {"<UNK>": 0}

        for words, _ in self.sentences:
            for word in words:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)

                for char in word:
                    if char not in self.char_to_ix:
                        self.char_to_ix[char] = len(self.char_to_ix)
        
        self.tag_to_ix = {}
        for _, tags in self.sentences:
            for tag in tags:
                if tag not in self.tag_to_ix:
                    self.tag_to_ix[tag] = len(self.tag_to_ix)
    
    def get_indices(self):
        if not 'word_to_ix' in self: self.create_indices()

        return (self.word_to_ix, self.char_to_ix, self.tag_to_ix)