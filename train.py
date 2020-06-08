import os
import time
from argparse import ArgumentParser

import pytorch_lightning as pl

from dataset import Embeddings, PerseusDataset
from model import LSTMCharTagger
from util import download_from_url

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="data")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=4e-3) # Calculated by the LR finder
    parser.add_argument('--word-lstm-hidden-dim', type=int, default=500)
    parser.add_argument('--disable-char-level', action='store_true')
    parser.add_argument('--disable-bidirectional', action='store_true')
    parser.add_argument('--num-lstm-layers', type=int, default=2)
    parser.add_argument('--char-lstm-hidden-dim', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--track', action='store_true')

    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    for dataset in ["train", "dev"]:
        if not os.path.isfile("%s/grc_perseus-ud-%s.conllu" % (hparams.data_dir, dataset)):
            download_from_url("https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master/grc_perseus-ud-%s.conllu" % dataset, "%s/grc_perseus-ud-%s.conllu" % (hparams.data_dir, dataset))

    print("Loading embeddings")
    embeddings = Embeddings(hparams.data_dir)
    print("Loading train dataset")
    train_data = PerseusDataset(hparams.data_dir + "/perseus-conllu/grc_perseus-ud-train.conllu", embeddings)
    print("Loading validation dataset")
    val_data = PerseusDataset(hparams.data_dir + "/perseus-conllu/grc_perseus-ud-dev.conllu", embeddings, tag_ids=train_data.tag_ids)

    model = LSTMCharTagger(hparams, train_data, val_data, embeddings.word_dim, embeddings.char_dim)

    pl.seed_everything(1)
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        logger=pl.loggers.WandbLogger(project="nlp_classics", log_model=False) if hparams.track else None,
        deterministic=True
    )
    trainer.fit(model)

    training_ckpt_path = "%s/%s.ckpt" % (hparams.data_dir, time.strftime("%Y%m%d-%H%M%S"))
    trainer.checkpoint_callback = None
    trainer.save_checkpoint(training_ckpt_path)
    print("Saved checkpoint to %s" % training_ckpt_path)
