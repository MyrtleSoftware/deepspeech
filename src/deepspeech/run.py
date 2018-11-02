import argparse
import logging
import os
import re

import torch
from torch.utils.data import DataLoader

from deepspeech.data.datasets import LibriSpeech
from deepspeech.data.loader import collate_input_sequences
from deepspeech.decoder import BeamCTCDecoder
from deepspeech.decoder import GreedyCTCDecoder
from deepspeech.global_state import GlobalState
from deepspeech.logging import LogLevelAction
from deepspeech.models import DeepSpeech
from deepspeech.models import DeepSpeech2


def main():
    args = get_parser().parse_args()

    global_state = GlobalState(exp_dir=args.exp_dir)
    global_state.log_frequency = args.slow_log_freq

    init_logger(args, global_state.exp_dir)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model = get_model(args, global_state.exp_dir)

    train_loader = get_train_loader(args, model)

    dev_loader = get_dev_loader(args, model)

    best_loss = float('inf')
    best_epoch = None

    for epoch in range(model.completed_epochs, args.n_epochs):
        model.train(train_loader)
        mean_loss = model.eval_loss(dev_loader)

        torch.save(model.state_dict(),
                   os.path.join(global_state.exp_dir, '%d.pt' % epoch))

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = epoch

    if best_epoch is not None:
        model.load_state_dict(torch.load(os.path.join(global_state.exp_dir,
                                                      '%d.pt' % best_epoch)))

    model.eval_wer(dev_loader)


def get_parser():
    parser = argparse.ArgumentParser(
        description='train and evaluate a DeepSpeech or DeepSpeech2 network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # logging -----------------------------------------------------------------

    parser.add_argument('--log_level',
                        action=LogLevelAction,
                        default='DEBUG',
                        help='logging level - see `logging` module')

    parser.add_argument('--slow_log_freq',
                        default=500,
                        type=int,
                        help='run slow logs every `slow_log_freq` batches')

    parser.add_argument('--exp_dir',
                        default=None,
                        help='path to directory to keep experimental data - '
                             'see `deepspeech.global_state.GlobalState`')

    parser.add_argument('--log_file',
                        default='log.txt',
                        help='filename to use for log file')

    # model -------------------------------------------------------------------

    parser.add_argument('model',
                        choices=['ds', 'ds2'],
                        help='model to train')

    parser.add_argument('--state_dict_path',
                        default=None,
                        help='path to initial state_dict to load into model - '
                             'takes precedence over '
                             '`--no_resume_from_exp_dir`')

    parser.add_argument('--no_resume_from_exp_dir',
                        action='store_true',
                        default=False,
                        help='do not load the last state_dict in exp_dir')

    # decoder -----------------------------------------------------------------

    parser.add_argument('--decoder',
                        default='beam',
                        choices=['beam', 'greedy'],
                        help='decoder to use - if equal to the model\'s '
                             'DEFAULT_DECODER_CLS then the model\'s '
                             'DEFAULT_DECODER_KWARGS are used for '
                             'initialisation. Other decoder-specific flags '
                             ' will (over)write entries if not None')

    parser.add_argument('--lm_path',
                        default=None,
                        help='path to language model - if None, no lm is used')

    parser.add_argument('--lm_weight',
                        default=None,
                        type=float,
                        help='language model weight in loss (i.e. alpha)')

    parser.add_argument('--word_weight',
                        default=None,
                        type=float,
                        help='word bonus weight in loss (i.e. beta)')

    parser.add_argument('--beam_width',
                        default=None,
                        type=int,
                        help='width of beam search')

    # optimizer ---------------------------------------------------------------

    parser.add_argument('--learning_rate',
                        default=0.0003,
                        type=float,
                        help='learning rate of Adam optimizer')

    # data --------------------------------------------------------------------

    parser.add_argument('--cachedir',
                        default='/tmp/data/cache/',
                        help='location to download dataset(s)')

    # training

    TRAIN_SUBSETS = ['train-clean-100',
                     'train-clean-360',
                     'train-other-500']
    parser.add_argument('--train_subsets',
                        default=TRAIN_SUBSETS,
                        choices=TRAIN_SUBSETS,
                        help='LibriSpeech subsets to train on',
                        nargs='+')

    parser.add_argument('--train_batch_size',
                        default=16,
                        type=int,
                        help='number of samples in a training batch')

    parser.add_argument('--train_num_workers',
                        default=4,
                        type=int,
                        help='number of subprocesses for train DataLoader')

    # validation

    parser.add_argument('--dev_batch_size',
                        default=16,
                        type=int,
                        help='number of samples in a validation batch')

    parser.add_argument('--dev_subsets',
                        default=['dev-clean', 'dev-other'],
                        choices=['dev-clean', 'dev-other',
                                 'test-clean', 'test-other'],
                        help='LibriSpeech subsets to evaluate loss and WER on',
                        nargs='+')

    parser.add_argument('--dev_num_workers',
                        default=4,
                        type=int,
                        help='number of subprocesses for dev DataLoader')

    # outer loop --------------------------------------------------------------

    parser.add_argument('--n_epochs',
                        default=15,
                        type=int,
                        help='number of epochs')

    # -------------------------------------------------------------------------

    return parser


def init_logger(args, exp_dir):
    logger = logging.getLogger()

    handler = logging.FileHandler(os.path.join(exp_dir, args.log_file))
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(funcName)s -'
                                      ' %(levelname)s: %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    logger.setLevel(logging.DEBUG)

    logger.debug(args)


def get_model(args, exp_dir):
    model_cls = {'ds': DeepSpeech, 'ds2': DeepSpeech2}[args.model]

    decoder_cls, decoder_kwargs = get_decoder(args, model_cls)

    model = model_cls(optimiser_cls=torch.optim.Adam,
                      optimiser_kwargs={'lr': args.learning_rate},
                      decoder_cls=decoder_cls,
                      decoder_kwargs=decoder_kwargs)

    state_dict_path = args.state_dict_path
    if state_dict_path is None and not args.no_resume_from_exp_dir:
        state_dict_path = _get_last_state_dict_path(exp_dir)
    if state_dict_path is not None:
        logger = logging.getLogger()
        logger.debug('restoring state_dict at %s' % state_dict_path)
        model.load_state_dict(torch.load(state_dict_path))

    return model


def get_decoder(args, model_cls):
    decoder_kwargs = {'alphabet': model_cls.ALPHABET,
                      'blank_symbol': model_cls.BLANK_SYMBOL}

    if args.decoder == 'beam':
        decoder_cls = BeamCTCDecoder

        if decoder_cls == model_cls.DEFAULT_DECODER_CLS:
            decoder_kwargs = model_cls.DEFAULT_DECODER_KWARGS

        if args.lm_weight is not None:
            decoder_kwargs['alpha'] = args.lm_weight
        if args.word_weight is not None:
            decoder_kwargs['beta'] = args.word_weight
        if args.beam_width is not None:
            decoder_kwargs['beam_width'] = args.beam_width
        if args.lm_path is not None:
            decoder_kwargs['model_path'] = args.lm_path

    elif args.decoder == 'greedy':
        decoder_cls = GreedyCTCDecoder

        if decoder_cls == model_cls.DEFAULT_DECODER_CLS:
            decoder_kwargs = model_cls.DEFAULT_DECODER_KWARGS

        beam_args = ['lm_weight', 'word_weight', 'beam_width', 'lm_path']
        for arg in beam_args:
            if getattr(args, arg) is not None:
                raise ValueError('greedy decoder selected but %r is not '
                                 'None' % arg)

    return decoder_cls, decoder_kwargs


def _get_last_state_dict_path(exp_dir):
    last_epoch = -1
    for f in os.listdir(exp_dir):
        match = re.match('([0-9]+).pt', f)
        if not match:
            continue
        epoch = int(match.groups()[0])
        if epoch > last_epoch:
            last_epoch = epoch
    if last_epoch == -1:
        return None
    return os.path.join(exp_dir, '%d.pt' % last_epoch)


def get_train_loader(args, model):
    train_cache = os.path.join(args.cachedir, 'train')
    train_dataset = LibriSpeech(root=train_cache,
                                subsets=args.train_subsets,
                                transform=model.transform,
                                target_transform=model.target_transform,
                                download=True)

    return DataLoader(train_dataset,
                      collate_fn=collate_input_sequences,
                      pin_memory=torch.cuda.is_available(),
                      num_workers=args.train_num_workers,
                      batch_size=args.train_batch_size,
                      shuffle=True)


def get_dev_loader(args, model):
    dev_cache = os.path.join(args.cachedir, 'dev')
    dev_dataset = LibriSpeech(root=dev_cache,
                              subsets=args.dev_subsets,
                              transform=model.transform,
                              target_transform=model.target_transform,
                              download=True)

    return DataLoader(dev_dataset,
                      collate_fn=collate_input_sequences,
                      pin_memory=torch.cuda.is_available(),
                      num_workers=args.dev_num_workers,
                      batch_size=args.dev_batch_size,
                      shuffle=False)
