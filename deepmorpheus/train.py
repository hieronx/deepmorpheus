import os
import time
from argparse import ArgumentParser

import pytorch_lightning as pl

from deepmorpheus.dataset import AncientGreekTrainDataset, AncientGreekValDataset, LatinTestDataset, LatinTrainDataset
from deepmorpheus.model import LSTMCharTagger
from deepmorpheus.util import download_from_url


def train(hparams):
    if hparams.language == 'ancient-greek':
        for dataset in ["train", "dev"]:
            if not os.path.isfile("%s/grc_perseus-ud-%s.conllu" % (hparams.data_dir, dataset)):
                download_from_url("https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master/grc_perseus-ud-%s.conllu" % dataset, "%s/grc_perseus-ud-%s.conllu" % (hparams.data_dir, dataset))

        train_data = AncientGreekTrainDataset(hparams.data_dir, hparams.language)
        val_data = AncientGreekValDataset(hparams.data_dir, hparams.language)
    elif hparams.language == 'latin':
        for dataset in ["train", "test"]:
            if not os.path.isfile("%s/la_perseus-ud-%s.conllu" % (hparams.data_dir, dataset)):
                download_from_url("https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-Perseus/master/la_perseus-ud-%s.conllu" % dataset, "%s/la_perseus-ud-%s.conllu" % (hparams.data_dir, dataset))

        train_data = LatinTrainDataset(hparams.data_dir, hparams.language)
        val_data = LatinTestDataset(hparams.data_dir, hparams.language) # TODO: we should not be using the test dataset for validation
    
    model = LSTMCharTagger(hparams, train_data.vocab, train_data, val_data)

    pl.seed_everything(1)
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        logger=pl.loggers.WandbLogger(project="deepmorpheus_%s" % hparams.language, log_model=False) if hparams.track else None,
        deterministic=True
    )
    trainer.fit(model)

    training_ckpt_path = "%s/%s-%s.ckpt" % (hparams.data_dir, hparams.language, time.strftime("%Y%m%d-%H%M%S"))
    trainer.checkpoint_callback = None
    trainer.save_checkpoint(training_ckpt_path)
    print("Saved checkpoint to %s" % training_ckpt_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="data")
    parser.add_argument('--language', choices=['ancient-greek', 'latin'], default="ancient-greek")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-3) # Calculated by the LR finder
    parser.add_argument('--word-embedding-dim', type=int, default=100)
    parser.add_argument('--word-lstm-hidden-dim', type=int, default=500)
    parser.add_argument('--disable-char-level', action='store_true')
    parser.add_argument('--disable-bidirectional', action='store_true')
    parser.add_argument('--num-lstm-layers', type=int, default=2)
    parser.add_argument('--char-embedding-dim', type=int, default=20)
    parser.add_argument('--char-lstm-hidden-dim', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--track', action='store_true')

    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    train(hparams)
