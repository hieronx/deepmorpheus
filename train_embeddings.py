import os
import re
from argparse import ArgumentParser
from multiprocessing import cpu_count

from betacode.conv import beta_to_uni as b2u
from bs4 import BeautifulSoup as Soup
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm


class EpochLogger(CallbackAny2Vec):
    def __init__(self, num_epochs, desc):
        self.num_epochs = num_epochs
        self.epoch = 0
        self.pbar = tqdm(total=num_epochs, desc=desc)

    def on_epoch_end(self, model):
        self.epoch += 1
        self.pbar.update(self.epoch)

        if self.epoch == self.num_epochs: self.pbar.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="data")
    parser.add_argument('--word-epochs', type=int, default=40)
    parser.add_argument('--char-epochs', type=int, default=40)
    parser.add_argument('--word-embedding-dim', type=int, default=100)
    parser.add_argument('--char-embedding-dim', type=int, default=50)
    args = parser.parse_args()

    if not os.path.isfile(args.data_dir + "/concat.txt"):
        def read_xml(dirpath, filename):
            url = os.path.join(dirpath, filename)
            soup = Soup(open(url, encoding='utf-8').read(), features='html.parser')
            output_file = open(output_filename, 'a', encoding='utf-8')
            for text in soup.findAll('text'):
                line = b2u(text.text.strip()).strip()
                if len(line) > 0:
                    output_file.write(line)
            output_file.close()
            
        # Prepare the data file by emptying it
        output_filename = args.data_dir + '/concat.txt'
        output_file = open(output_filename, 'w')
        output_file.write('')
        output_file.close()

        for (dirpath, dirnames, filenames) in os.walk(args.data_dir + '/greek_texts/'):
            for filename in tqdm(filenames, total=len(filenames), desc='Parsing'):
                read_xml(dirpath, filename)

    word_lines = []
    with open(args.data_dir + '/concat.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, total=len(lines), desc='Loading text data'):
            line = re.sub(r'[.,·;]', '', line)
            line = re.sub(r'\s+', ' ', line)
            word_lines.append(line.split(' '))

    word_logger = EpochLogger(num_epochs=args.word_epochs, desc="Training word embeddings")
    word_model = Word2Vec(word_lines, min_count=3, size=args.word_embedding_dim, workers=cpu_count(), iter=args.word_epochs, callbacks=[word_logger])
    word_model.wv.save_word2vec_format(args.data_dir + '/word_embeddings.bin')
    print("Saved word embeddings to %s" % (args.data_dir + '/word_embeddings.bin'))

    char_lines = []
    for line in tqdm(word_lines, total=len(word_lines), desc='Splitting into characters'):
        sentence_chars = []
        for word in line:
            sentence_chars += list(word)
        char_lines.append(sentence_chars)

    char_logger = EpochLogger(num_epochs=args.char_epochs, desc="Training character embeddings")
    char_model = Word2Vec(char_lines, size=args.char_embedding_dim, workers=cpu_count(), iter=args.char_epochs, callbacks=[char_logger])
    char_model.wv.save_word2vec_format(args.data_dir + '/char_embeddings.bin')
    print("Saved character embeddings to %s" % (args.data_dir + '/char_embeddings.bin'))
