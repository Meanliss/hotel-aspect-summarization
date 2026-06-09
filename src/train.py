import argparse
from contextlib import nullcontext
import json
import os.path
from random import seed

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch.utils.tensorboard as tb
except ImportError:
    class _NoOpSummaryWriter(object):
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

    class _NoOpTensorboard(object):
        SummaryWriter = _NoOpSummaryWriter

    tb = _NoOpTensorboard()
from scipy.cluster.vq import kmeans
from torch.utils.data import DataLoader

from semae import SemanticAutoencoderModel
from utils.data import *
from utils.training import *


# parts of the code has been
# adapted from: https://github.com/stangelid/qt

def tensor_summary(tensor):
    tensor = tensor.detach()
    finite = torch.isfinite(tensor)
    finite_count = int(finite.sum().item())
    total = tensor.numel()
    if finite_count == 0:
        return {
            'finite': 0,
            'total': total,
            'min': None,
            'max': None,
            'mean_abs': None,
        }
    values = tensor[finite]
    return {
        'finite': finite_count,
        'total': total,
        'min': float(values.min().item()),
        'max': float(values.max().item()),
        'mean_abs': float(values.abs().mean().item()),
    }


def first_bad_parameter(model):
    for name, param in model.named_parameters():
        tensors = []
        if param is not None:
            tensors.append(('param', param.data))
        if param.grad is not None:
            tensors.append(('grad', param.grad.data))
        for kind, tensor in tensors:
            finite = torch.isfinite(tensor)
            if not bool(finite.all().item()):
                bad = int((~finite).sum().item())
                return name, kind, bad, tensor_summary(tensor)
    return None


def grad_report(model):
    report = {
        'bad_tensors': 0,
        'bad_elements': 0,
        'first_bad': None,
        'max_abs_grad': 0.0,
        'robust_norm': 0.0,
        'grad_tensors': 0,
    }
    sq_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        report['grad_tensors'] += 1
        grad = param.grad.detach()
        finite = torch.isfinite(grad)
        if not bool(finite.all().item()):
            bad = int((~finite).sum().item())
            report['bad_tensors'] += 1
            report['bad_elements'] += bad
            if report['first_bad'] is None:
                report['first_bad'] = {
                    'name': name,
                    'bad_elements': bad,
                    'summary': tensor_summary(grad),
                }
            grad = torch.where(finite, grad, torch.zeros_like(grad))
        if grad.numel() == 0:
            continue
        max_abs = float(grad.abs().max().item())
        if max_abs > report['max_abs_grad']:
            report['max_abs_grad'] = max_abs
        param_norm = float(torch.linalg.vector_norm(grad.float(), ord=2).item())
        if np.isfinite(param_norm):
            sq_norm += param_norm * param_norm
    report['robust_norm'] = float(np.sqrt(sq_norm))
    return report


def zero_bad_grad_values(model):
    for param in model.parameters():
        if param.grad is None:
            continue
        finite = torch.isfinite(param.grad)
        if not bool(finite.all().item()):
            param.grad.data = torch.where(finite, param.grad.data,
                                          torch.zeros_like(param.grad.data))


def clip_gradients(model, max_norm, report):
    if max_norm <= 0.0:
        return 0.0
    if report['bad_elements'] > 0:
        return report['robust_norm']
    norm = report['robust_norm']
    if norm > max_norm and norm > 0.0:
        scale = max_norm / (norm + 1e-12)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(scale)
    return norm


def batch_diagnostics(src, tgt, gld, full_ids):
    src_ne_pad = src != 0
    sent_lengths = src_ne_pad.sum(dim=2)
    review_lengths = (sent_lengths > 0).sum(dim=1)
    return {
        'ids': list(full_ids) if full_ids is not None else [],
        'batch_size': int(src.size(0)),
        'max_sentences': int(src.size(1)),
        'max_src_tokens': int(src.size(2)),
        'max_tgt_tokens': int(tgt.size(2)),
        'non_padding_tokens': int((tgt != 0).sum().item()),
        'review_lengths': [int(x) for x in review_lengths.detach().cpu()],
        'max_sentence_tokens': int(sent_lengths.max().item()),
        'zero_sentence_slots': int((sent_lengths == 0).sum().item()),
    }


def log_nonfinite(prefix, epoch, batch_idx, loss, g_loss, q_loss, src, tgt, gld,
                  full_ids, extra=None):
    print('NONFINITE_DIAGNOSTIC {0} epoch={1} batch={2}'.format(
        prefix, epoch + 1, batch_idx))
    print('  loss={0} g_loss={1} q_loss={2}'.format(
        float(loss.detach().item()) if torch.is_tensor(loss) and loss.numel() == 1 else loss,
        float(g_loss.detach().item()) if torch.is_tensor(g_loss) and g_loss.numel() == 1 else g_loss,
        float(q_loss.detach().item()) if torch.is_tensor(q_loss) and q_loss.numel() == 1 else q_loss))
    print('  batch={0}'.format(batch_diagnostics(src, tgt, gld, full_ids)))
    if extra is not None:
        print('  extra={0}'.format(extra))


def finite_tensor_info(name, tensor):
    summary = tensor_summary(tensor)
    return '{0}: finite={1}/{2} min={3} max={4} mean_abs={5}'.format(
        name, summary['finite'], summary['total'], summary['min'],
        summary['max'], summary['mean_abs'])


def parse_epoch_batch_limits(value):
    limits = {}
    if not value:
        return limits
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        epoch_text, batch_text = item.split(':', 1)
        epoch = int(epoch_text)
        batch = int(batch_text)
        if epoch <= 0 or batch <= 0:
            raise ValueError('Epoch and batch limits must be positive: {0}'.
                             format(item))
        limits[epoch] = batch
    return limits


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)

    data_arg_group = argparser.add_argument_group('Data arguments')
    data_arg_group.add_argument('--data',
                                help='training data in json format',
                                type=str,
                                default='../data/space/json/space_train.json')
    data_arg_group.add_argument(
        '--sentencepiece',
        help='sentencepiece model file',
        type=str,
        default='../data/sentencepiece/spm_unigram_32k.model')
    data_arg_group.add_argument(
        '--max_num_entities',
        help='maximum number of entities to load for training (default: all)',
        type=int,
        default=None)
    data_arg_group.add_argument(
        '--max_rev_len',
        help='maximum number of sentences per review (default: 40)',
        type=int,
        default=40)
    data_arg_group.add_argument(
        '--max_sen_len',
        help='maximum number of tokens per sentence (default: 40)',
        type=int,
        default=40)

    model_arg_group = argparser.add_argument_group('Model hyperparams')
    model_arg_group.add_argument('--d_model',
                                 help='model dimensionality (default: 320)',
                                 type=int,
                                 default=320)
    model_arg_group.add_argument(
        '--codebook_size',
        help='size of quantization codebook (default: 1024)',
        type=int,
        default=1024)
    model_arg_group.add_argument(
        '--output_nheads',
        help='number of output sentence heads (default: 8)',
        type=int,
        default=8)
    model_arg_group.add_argument(
        '--nlayers',
        help='number of sentence-level layers (default: 3)',
        type=int,
        default=3)
    model_arg_group.add_argument('--internal_nheads',
                                 help='number of attention heads (default: 4)',
                                 type=int,
                                 default=4)
    model_arg_group.add_argument(
        '--d_ff',
        help='feed-forward dimensionality (default: 512)',
        type=int,
        default=512)
    model_arg_group.add_argument('--in_pos',
                                 help='use input positional embeddings',
                                 action='store_true')
    model_arg_group.add_argument('--out_pos',
                                 help='use output positional embeddings',
                                 action='store_true')
    model_arg_group.add_argument(
        '--dropout',
        help='transformer dropout probability (default: 0.0)',
        type=float,
        default=0.0)

    train_arg_group = argparser.add_argument_group(
        'Basic training hyperparams')
    train_arg_group.add_argument('--batch_size',
                                 help='the batch size (default: 5)',
                                 type=int,
                                 default=5)
    train_arg_group.add_argument('--epochs',
                                 help='number of epochs (default: 20)',
                                 type=int,
                                 default=10)
    train_arg_group.add_argument('--lr',
                                 help='initial learning rate',
                                 type=float,
                                 default=0.001)
    train_arg_group.add_argument('--lr_decay',
                                 help='learning rate decay (default: 0.9)',
                                 type=float,
                                 default=0.9)
    train_arg_group.add_argument('--label_smoothing',
                                 help='label smoothing coeff (default: 0.1)',
                                 type=float,
                                 default=0.1)
    train_arg_group.add_argument(
        '--commitment_cost',
        help='VQ-VAE commitment coefficient (default: 1.00)',
        type=float,
        default=1.00)

    train_arg_group.add_argument(
        '--l1_cost',
        help='VQ-VAE commitment coefficient (default: 1000.00)',
        type=float,
        default=1000.00)
    train_arg_group.add_argument(
        '--entropy_cost',
        help='VQ-VAE commitment coefficient (default: 0.00005)',
        type=float,
        default=0.00005)
    train_arg_group.add_argument(
        '--grad_clip',
        help='clip gradient norm to this value; disabled if <= 0 (default: 0)',
        type=float,
        default=0.0)
    train_arg_group.add_argument(
        '--skip_nonfinite',
        help='compat alias for --nonfinite_policy skip',
        action='store_true')
    train_arg_group.add_argument(
        '--diagnose_nonfinite',
        help='log loss, batch, parameter, and gradient diagnostics',
        action='store_true')
    train_arg_group.add_argument(
        '--nonfinite_policy',
        help='what to do when loss/gradients contain NaN/Inf (default: fail)',
        choices=['fail', 'skip', 'zero_bad_grads'],
        default='fail')
    train_arg_group.add_argument(
        '--safe_normalize',
        help='clamp quantizer normalization denominators by epsilon',
        action='store_true')
    train_arg_group.add_argument(
        '--normalize_epsilon',
        help='epsilon used by --safe_normalize (default: 1e-5)',
        type=float,
        default=1e-5)
    train_arg_group.add_argument(
        '--max_train_batches',
        help='optional per-epoch train batch limit for diagnostics',
        type=int,
        default=None)
    train_arg_group.add_argument(
        '--max_train_batches_by_epoch',
        help='comma-separated 1-indexed limits like 1:11061,2:1409',
        default='')
    train_arg_group.add_argument(
        '--detect_anomaly_batch',
        help='enable torch autograd anomaly detection for this 1-indexed train batch',
        type=int,
        default=None)
    train_arg_group.add_argument(
        '--detect_anomaly_epoch',
        help='epoch used with --detect_anomaly_batch (1-indexed, default: any epoch)',
        type=int,
        default=None)
    train_arg_group.add_argument(
        '--bool_attn_masks',
        help='use boolean causal attention masks to match PyTorch padding mask dtype',
        action='store_true')
    train_arg_group.add_argument(
        '--disable_tf32',
        help='disable TF32 matmul/cudnn kernels for numerical diagnostics',
        action='store_true')

    train_arg_group = argparser.add_argument_group('Soft EMA hyperparams')
    train_arg_group.add_argument(
        '--ema_temp',
        help=
        'sampling temperature for Soft EMA codebook training (default: 1.0)',
        type=float,
        default=1.0)
    train_arg_group.add_argument(
        '--ema_num_samples',
        help='number of samples for Soft EMA codebook training (default: 10)',
        type=int,
        default=10)
    train_arg_group.add_argument(
        '--ema_decay',
        help='exponential decay for EMA (default: 0.99)',
        type=float,
        default=0.99)


    lr_arg_group = argparser.add_argument_group('Learning rate drop-off hyperparams',
            'Learning rate drop-off reduces the lr to 0 after some epochs ' + \
            'and slowly increases it again. May help with quantization collapse, ' + \
            'but not necessary in most cases.')
    lr_arg_group.add_argument(
        '--lr_drop_enc',
        help='drop lr for encoder to zero and increase slowly',
        action='store_true')
    lr_arg_group.add_argument(
        '--lr_drop_all',
        help='drop lr for all to zero and increase slowly',
        action='store_true')
    lr_arg_group.add_argument('--lr_drop_epoch',
                              help='epoch to drop learning rate to zero',
                              type=int,
                              default=-1)
    lr_arg_group.add_argument(
        '--lr_rtrn_epochs',
        help='number of epochs to increase learning rate to normal after drop',
        type=int,
        default=-1)

    warmup_arg_group = argparser.add_argument_group('Transformer warmup hyperparams',
            'With transformer warmup, QT is trained without quantization for ' + \
            'some epochs, and then gradually introduces quantization. Improves ' + \
            'training stability.')
    warmup_arg_group.add_argument(
        '--no_transformer_warmup',
        help='disable transformer warmup before quantization',
        action='store_true')
    warmup_arg_group.add_argument(
        '--warmup_epochs',
        help='don\'t quantize at all for this many epochs (default: 4)',
        type=int,
        default=4)
    warmup_arg_group.add_argument(
        '--no_warmup_annealing',
        help='disable slow decrease of non-quantized residual coefficient',
        action='store_true')
    warmup_arg_group.add_argument(
        '--warmup_annealing_min',
        help=
        'minimum residual coefficient for non-quantized path (default: 0.0)',
        type=float,
        default=0.0)
    warmup_arg_group.add_argument(
        '--warmup_annealing_epochs',
        help=
        'non-quantized residual reduction lasts this many epochs (default: 2)',
        type=int,
        default=2)

    kmeans_arg_group = argparser.add_argument_group(
        'K-means initialization hyperparams',
        'Initialize codebook with kmeans after transformer warmup')
    kmeans_arg_group.add_argument(
        '--no_kmeans',
        help='disable kmeans codebook initialization after warmup',
        action='store_true')
    kmeans_arg_group.add_argument(
        '--kmeans_batches',
        help='number of batches for kmeans (default: 100)',
        type=int,
        default=100)
    kmeans_arg_group.add_argument(
        '--kmeans_iter',
        help='number of iterations for kmeans (default: 50)',
        type=int,
        default=50)
    kmeans_arg_group.add_argument(
        '--kmeans_bad_vector_policy',
        help='what to do when K-means input has zero/non-finite vectors',
        choices=['fail', 'filter'],
        default='fail')
    kmeans_arg_group.add_argument(
        '--kmeans_max_bad_vectors',
        help='max zero/non-finite K-means vectors allowed when filtering',
        type=int,
        default=0)

    other_arg_group = argparser.add_argument_group('Other arguments')
    other_arg_group.add_argument(
        '--run_id',
        help='unique run id (for logging and saved models)',
        type=str,
        default='run1')
    other_arg_group.add_argument('--gpu',
                                 help='gpu device to use (default: use cpu)',
                                 type=int,
                                 default=-1)
    other_arg_group.add_argument(
        '--logdir',
        help='directory to put tensorboard logs (default: \'../logs\')',
        type=str,
        default='../logs')
    other_arg_group.add_argument(
        '--log_every',
        help='log every n forward passes (default: 50)',
        type=int,
        default=50)
    other_arg_group.add_argument(
        '--savedir',
        help='directory to put saved model snapshots (default: \'../models\')',
        type=str,
        default='../models')
    other_arg_group.add_argument(
        '--save_every',
        help=
        'save model snapshot every N epochs (default: save on every epoch)',
        type=int,
        default=1)
    other_arg_group.add_argument('--seed',
                                 help='random seed',
                                 type=int,
                                 default=1)
    other_arg_group.add_argument(
        '--data_seed',
        help=
        'random seed for dataset (only affects batching and entity subsampling)',
        type=int,
        default=1)
    other_arg_group.add_argument(
        '--no_eval',
        help='skip dev/test evaluation after train epoch',
        action='store_true')
    other_arg_group.add_argument(
        '--resume_model',
        help='load a saved model snapshot before training',
        type=str,
        default='')
    other_arg_group.add_argument(
        '--start_epoch',
        help='zero-based epoch index to resume from',
        type=int,
        default=0)

    args = argparser.parse_args()
    if args.skip_nonfinite:
        args.nonfinite_policy = 'skip'
    max_train_batches_by_epoch = parse_epoch_batch_limits(
        args.max_train_batches_by_epoch)

    seed(args.data_seed)
    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision('highest')
        except AttributeError:
            pass

    if args.gpu >= 0:
        device = torch.device('cuda:{0}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    data_path = args.data
    spm_path = args.sentencepiece
    save_path = args.savedir
    log_path = args.logdir
    os.makedirs(save_path, exist_ok=True)
    if log_path != '':
        os.makedirs(log_path, exist_ok=True)

    # read data from json file
    print("Reading data ...")
    f = open(data_path, 'r', encoding='utf-8')
    data = json.load(f)
    f.close()
    print("Done!")

    # initialize dataset
    dataset = ReviewDataset(data,
                            sample_size=args.max_num_entities,
                            spmodel=spm_path,
                            max_sen_len=args.max_sen_len,
                            max_rev_len=args.max_rev_len)
    vocab_size = dataset.vocab_size
    nclasses = dataset.nclasses

    # prepare train/dev/test splits
    dataset.split()

    # samplers for each split
    train_sampler = \
            ReviewBucketBatchSampler(dataset, args.batch_size, split='train')
    dev_sampler = \
            ReviewBucketBatchSampler(dataset, args.batch_size, split='dev')
    test_sampler = \
            ReviewBucketBatchSampler(dataset, args.batch_size, split='test')

    # wrapper for collate function
    collator = ReviewCollator(padding_idx=dataset.pad_id(),
                              unk_idx=dataset.unk_id(),
                              bos_idx=dataset.bos_id(),
                              eos_idx=dataset.eos_id())

    # one dataloader per split
    train_collate_fn = collator.collate_reviews_generation
    if args.diagnose_nonfinite:
        train_collate_fn = collator.collate_reviews_generation_with_ids
    train_dl = DataLoader(dataset,
                          batch_sampler=train_sampler,
                          collate_fn=train_collate_fn)
    dev_dl = DataLoader(dataset,
                        batch_sampler=dev_sampler,
                        collate_fn=collator.collate_reviews_generation)
    test_dl = DataLoader(dataset,
                         batch_sampler=test_sampler,
                         collate_fn=collator.collate_reviews_generation)
    nbatches_trn = len(train_dl)
    nbatches_dev = len(dev_dl)
    nbatches_tst = len(test_dl)

    pad_id = dataset.pad_id()
    bos_id = dataset.bos_id()
    eos_id = dataset.eos_id()
    unk_id = dataset.unk_id()

    torch.manual_seed(args.seed)

    # define model
    model = SemanticAutoencoderModel(vocab_size,
                                     d_model=args.d_model,
                                     temp=args.ema_temp,
                                     num_samples=args.ema_num_samples,
                                     codebook_size=args.codebook_size,
                                     l1_cost=args.l1_cost,
                                     entropy_cost=args.entropy_cost,
                                     nlayers=args.nlayers,
                                     internal_nheads=args.internal_nheads,
                                     output_nheads=args.output_nheads,
                                     d_ff=args.d_ff,
                                     use_in_pos=args.in_pos,
                                     use_out_pos=args.out_pos,
                                     use_bool_attention_masks=args.bool_attn_masks,
                                     ema_decay=args.ema_decay,
                                     epsilon=args.normalize_epsilon,
                                     safe_normalize=args.safe_normalize,
                                     dropout=args.dropout)

    model.to(device)
    if args.resume_model != '':
        print('Loading resume model: {0}'.format(args.resume_model))
        model = torch.load(args.resume_model, map_location=device, weights_only=False)
        model.to(device)

    # prepare optimizer and learning rate scheduler
    if args.lr_drop_all:
        optimizer = \
            torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_calc)
    elif args.lr_drop_enc:
        param_groups = \
                [
                    {'params': model.in_emb.parameters()},
                    {'params': model.encoder.parameters()},
                    {'params': model.decoder.parameters()},
                    {'params': model.linear.parameters()}
                ]
        lambda1 = lr_calc
        lambda2 = lambda epoch: args.lr_decay**epoch
        lr_lambdas = [lambda1, lambda1, lambda2, lambda2]
        optimizer = \
            torch.optim.Adam(param_groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas)
    else:
        optimizer = \
            torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, args.lr_decay)

    # define losses
    if args.label_smoothing == 0.0:
        criterion = \
                nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
    else:
        criterion = \
                LabelSmoothingLoss(args.label_smoothing, vocab_size, ignore_index=pad_id)
    valid_criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')

    criterion = criterion.to(device)
    valid_criterion = valid_criterion.to(device)

    # prepare transformer warmup scheduler
    if (not args.no_transformer_warmup) and (not args.no_warmup_annealing):
        warmup_scheduler = \
                ResidualCoefficientScheduler(args.warmup_epochs,
                        args.warmup_annealing_epochs, nbatches_trn,
                        min_coeff=args.warmup_annealing_min)

    if args.logdir != '':
        tb_writer = tb.SummaryWriter(os.path.join(log_path, args.run_id))

    for epoch in range(args.start_epoch, args.epochs):
        train_batch_limit = max_train_batches_by_epoch.get(
            epoch + 1, args.max_train_batches)
        # initialize loss and counts for accuracy
        running_loss = 0.0
        running_g_loss = 0.0
        running_q_loss = 0.0
        epoch_stats = {
            'finite_batches': 0,
            'skipped_batches': 0,
            'bad_loss_batches': 0,
            'bad_grad_batches': 0,
            'zeroed_bad_grad_batches': 0,
            'max_grad_norm': 0.0,
            'max_abs_grad': 0.0,
            'first_bad': None,
        }
        model.train()

        # quantize or not
        quantize = (args.no_transformer_warmup or epoch >= args.warmup_epochs)

        # if warmup is over, initialize codebook with kmeans
        if (not args.no_kmeans) and epoch == args.warmup_epochs:
            with torch.no_grad():
                model.eval()
                sentence_vecs = []
                for i, batch in enumerate(train_dl):
                    if i == args.kmeans_batches:
                        break

                    if args.diagnose_nonfinite:
                        src, tgt, gld, _ = batch
                    else:
                        src, tgt, gld = batch
                    src, tgt, gld = [x.to(device) for x in (src, tgt, gld)]
                    out, _ = model.encode(src, quantize=False)
                    sentence_vecs.append(
                        out.reshape(-1, args.d_model).detach().to('cpu'))
                sentence_vecs = torch.cat(sentence_vecs,
                                          dim=0).detach().numpy()

                length = np.sqrt((sentence_vecs**2).sum(axis=1))[:, None]
                valid = np.isfinite(sentence_vecs).all(axis=1) \
                    & np.isfinite(length[:, 0]) & (length[:, 0] > 0)
                bad_vectors = int((~valid).sum())
                if bad_vectors > 0:
                    print('KMEANS_VECTOR_DIAGNOSTIC total={0} bad={1} policy={2}'.
                          format(sentence_vecs.shape[0], bad_vectors,
                                 args.kmeans_bad_vector_policy))
                    if args.kmeans_bad_vector_policy == 'fail':
                        raise ValueError(
                            'K-means input has {0} zero/non-finite vectors'.
                            format(bad_vectors))
                    if bad_vectors > args.kmeans_max_bad_vectors:
                        raise ValueError(
                            'K-means bad vector count {0} exceeds limit {1}'.
                            format(bad_vectors, args.kmeans_max_bad_vectors))
                    sentence_vecs = sentence_vecs[valid]
                    length = length[valid]
                if sentence_vecs.shape[0] < args.codebook_size:
                    raise ValueError(
                        'Not enough finite vectors for K-means: {0} < {1}'.
                        format(sentence_vecs.shape[0], args.codebook_size))
                sentence_vecs = sentence_vecs / length

                kmeans_codebook, _ = kmeans(sentence_vecs,
                                            args.codebook_size,
                                            iter=args.kmeans_iter)

                # in case of missing clusters, fill in random ones.
                # missing cluster may occur when there are identical
                # sentence vectors in the clustered data
                if kmeans_codebook.shape[0] < args.codebook_size:
                    num_missing_clusters = args.codebook_size - kmeans_codebook.shape[
                        0]
                    new_clusters = np.random.randn(num_missing_clusters,
                                                   args.d_model)
                    kmeans_codebook = np.concatenate(
                        (kmeans_codebook, new_clusters), axis=0)

                model.encoder.set_codebook(torch.Tensor(kmeans_codebook))
                model.train()

            # save model snapshot
            if args.save_every is not None:
                torch.save(
                    model,
                    '{0}/{1}_{2}pkm_model.pt'.format(save_path, args.run_id,
                                                     epoch))

        train_dataloader = tqdm(train_dl)
        for i, batch in enumerate(train_dataloader, 1):
            full_ids = None
            if args.diagnose_nonfinite:
                src, tgt, gld, full_ids = batch
            else:
                src, tgt, gld = batch
            src, tgt, gld = [x.to(device) for x in (src, tgt, gld)]
            batch_size, nsent, src_ntokens = src.size()

            optimizer.zero_grad()

            if not args.no_transformer_warmup:
                if not args.no_warmup_annealing:
                    residual_coeff = warmup_scheduler.get_residual_coefficient(
                        i, epoch)
                else:
                    residual_coeff = 0.0 if quantize else 1.0
            else:
                residual_coeff = 0.0

            anomaly_enabled = (
                args.detect_anomaly_batch is not None
                and i == args.detect_anomaly_batch
                and (args.detect_anomaly_epoch is None
                     or epoch + 1 == args.detect_anomaly_epoch))
            if anomaly_enabled:
                print('ANOMALY_DIAGNOSTIC enabled epoch={0} batch={1}'.
                      format(epoch + 1, i))

            context = torch.autograd.detect_anomaly(check_nan=True) \
                if anomaly_enabled else nullcontext()
            with context:
                logits, q_loss = \
                        model(src, tgt, quantize=quantize, residual_coeff=residual_coeff)
                if anomaly_enabled:
                    print('  {0}'.format(finite_tensor_info('logits', logits)))
                    print('  {0}'.format(finite_tensor_info('q_loss', q_loss if torch.is_tensor(q_loss) else torch.tensor(q_loss))))
                out = logits
                if args.label_smoothing > 0.0:
                    out = F.log_softmax(out, dim=-1)
                    if anomaly_enabled:
                        print('  {0}'.format(
                            finite_tensor_info('log_softmax', out)))

                g_loss = criterion(out.flatten(end_dim=-2), gld.flatten())
                non_padding_elem = (tgt != pad_id).sum().item()
                g_loss /= batch_size * nsent
                q_loss *= float(non_padding_elem) / (batch_size * nsent)

                loss = g_loss + q_loss

                if not torch.isfinite(loss):
                    epoch_stats['bad_loss_batches'] += 1
                    if epoch_stats['first_bad'] is None:
                        epoch_stats['first_bad'] = 'loss epoch={0} batch={1}'.format(
                            epoch + 1, i)
                    if args.diagnose_nonfinite:
                        log_nonfinite('loss', epoch, i, loss, g_loss, q_loss,
                                      src, tgt, gld, full_ids)
                    if args.nonfinite_policy == 'skip':
                        epoch_stats['skipped_batches'] += 1
                        continue
                    raise RuntimeError(
                        'Non-finite loss at epoch {0}, batch {1}'.format(
                            epoch + 1, i))

                loss.backward()
            report = grad_report(model)
            if report['robust_norm'] > epoch_stats['max_grad_norm']:
                epoch_stats['max_grad_norm'] = report['robust_norm']
            if report['max_abs_grad'] > epoch_stats['max_abs_grad']:
                epoch_stats['max_abs_grad'] = report['max_abs_grad']

            if report['bad_elements'] > 0:
                epoch_stats['bad_grad_batches'] += 1
                if epoch_stats['first_bad'] is None:
                    epoch_stats['first_bad'] = 'grad epoch={0} batch={1} {2}'\
                        .format(epoch + 1, i, report['first_bad'])
                if args.diagnose_nonfinite:
                    log_nonfinite('grad', epoch, i, loss, g_loss, q_loss, src,
                                  tgt, gld, full_ids, extra=report)
                if args.nonfinite_policy == 'skip':
                    epoch_stats['skipped_batches'] += 1
                    optimizer.zero_grad()
                    continue
                if args.nonfinite_policy == 'zero_bad_grads':
                    zero_bad_grad_values(model)
                    epoch_stats['zeroed_bad_grad_batches'] += 1
                    report = grad_report(model)
                else:
                    raise RuntimeError(
                        'Non-finite gradients at epoch {0}, batch {1}: {2}'.
                        format(epoch + 1, i, report['first_bad']))

            if args.grad_clip > 0.0:
                clip_gradients(model, args.grad_clip, report)
            optimizer.step()
            epoch_stats['finite_batches'] += 1

            running_loss += loss.item()
            running_g_loss += g_loss.item()
            if quantize:
                running_q_loss += q_loss.item()

            train_dataloader.set_description(
                "Loss: {} G Loss: {} Q Loss: {}".format(
                    running_loss / (i + 1), running_g_loss / (i + 1),
                    running_q_loss / (i + 1)))

            # log average loss per batch every k passes
            if args.logdir != '' and i % args.log_every == args.log_every - 1:
                step = epoch * nbatches_trn + i
                running_uq_loss = running_q_loss / args.commitment_cost
                lrs = lr_scheduler.get_lr()
                lr_enc = lrs[0]
                if len(lrs) > 1:
                    lr_dec = lrs[2]
                else:
                    lr_dec = lr_enc
                tb_writer.add_scalar('loss/train',
                                     running_loss / args.log_every, step)
                tb_writer.add_scalar('g_loss/train',
                                     running_g_loss / args.log_every, step)
                tb_writer.add_scalar('q_loss/train',
                                     running_q_loss / args.log_every, step)
                tb_writer.add_scalar('uq_loss/train',
                                     running_uq_loss / args.log_every, step)
                tb_writer.add_scalar('residual_coeff/train', residual_coeff,
                                     step)
                tb_writer.add_scalar('learning_rate/enc', lr_enc, step)
                tb_writer.add_scalar('learning_rate/dec', lr_dec, step)
                running_loss = 0.0
                running_g_loss = 0.0
                running_q_loss = 0.0

            if train_batch_limit is not None and i >= train_batch_limit:
                print('Stopping train epoch {0} early at diagnostic batch limit {1}'.
                      format(epoch + 1, train_batch_limit))
                break

        print('EPOCH_DIAGNOSTIC epoch={0} finite_batches={1} skipped_batches={2} '
              'bad_loss_batches={3} bad_grad_batches={4} zeroed_bad_grad_batches={5} '
              'max_grad_norm={6:.6g} max_abs_grad={7:.6g} first_bad={8}'.
              format(epoch + 1, epoch_stats['finite_batches'],
                     epoch_stats['skipped_batches'],
                     epoch_stats['bad_loss_batches'],
                     epoch_stats['bad_grad_batches'],
                     epoch_stats['zeroed_bad_grad_batches'],
                     epoch_stats['max_grad_norm'],
                     epoch_stats['max_abs_grad'], epoch_stats['first_bad']))

        if not args.no_eval:
            with torch.no_grad():
            # initialize loss
                running_loss = 0.0
                running_g_loss = 0.0
                running_q_loss = 0.0
                model.eval()
                for i, batch in enumerate(dev_dl):
                    if args.diagnose_nonfinite:
                        src, tgt, gld, _ = batch
                    else:
                        src, tgt, gld = batch
                    src, tgt, gld = [x.to(device) for x in (src, tgt, gld)]
                    batch_size, nsent, src_ntokens = src.size()

                    out, q_loss = \
                            model(src, tgt, quantize=quantize, residual_coeff=residual_coeff)
                    g_loss = valid_criterion(out.flatten(end_dim=-2),
                                             gld.flatten())

                    non_padding_elem = (tgt != pad_id).sum().item()
                    g_loss /= batch_size * nsent
                    q_loss *= float(non_padding_elem) / (batch_size * nsent)

                    loss = g_loss + q_loss

                    running_loss += loss.item()
                    running_g_loss += g_loss.item()

                    # log average loss per batch every k passes
                    if args.logdir != '' and i % args.log_every == args.log_every - 1:
                        step = epoch * nbatches_dev + i
                        running_uq_loss = running_q_loss / args.commitment_cost
                        tb_writer.add_scalar('loss/dev',
                                             running_loss / args.log_every, step)
                        tb_writer.add_scalar('g_loss/dev',
                                             running_g_loss / args.log_every, step)
                        tb_writer.add_scalar('q_loss/dev',
                                             running_q_loss / args.log_every, step)
                        tb_writer.add_scalar('uq_loss/dev',
                                             running_uq_loss / args.log_every,
                                             step)
                        tb_writer.add_scalar('residual_coeff/dev', residual_coeff,
                                             step)
                        running_loss = 0.0
                        running_g_loss = 0.0
                        running_q_loss = 0.0

            # initialize loss
                running_loss = 0.0
                running_g_loss = 0.0
                running_q_loss = 0.0
                model.eval()
                for i, batch in enumerate(test_dl):
                    if args.diagnose_nonfinite:
                        src, tgt, gld, _ = batch
                    else:
                        src, tgt, gld = batch
                    src, tgt, gld = [x.to(device) for x in (src, tgt, gld)]
                    batch_size, nsent, src_ntokens = src.size()

                    out, q_loss = \
                            model(src, tgt, quantize=quantize, residual_coeff=residual_coeff)
                    g_loss = valid_criterion(out.flatten(end_dim=-2),
                                             gld.flatten())

                    non_padding_elem = (tgt != pad_id).sum().item()
                    g_loss /= batch_size * nsent
                    q_loss *= float(non_padding_elem) / (batch_size * nsent)

                    loss = g_loss + q_loss

                    running_loss += loss.item()
                    running_g_loss += g_loss.item()
                    if quantize:
                        running_q_loss += q_loss.item()

                    # log average loss per batch every k passes
                    if args.logdir != '' and i % args.log_every == args.log_every - 1:
                        step = epoch * nbatches_dev + i
                        running_uq_loss = running_q_loss / args.commitment_cost
                        tb_writer.add_scalar('loss/test',
                                             running_loss / args.log_every, step)
                        tb_writer.add_scalar('g_loss/test',
                                             running_g_loss / args.log_every, step)
                        tb_writer.add_scalar('q_loss/test',
                                             running_q_loss / args.log_every, step)
                        tb_writer.add_scalar('uq_loss/test',
                                             running_uq_loss / args.log_every,
                                             step)
                        tb_writer.add_scalar('residual_coeff/test', residual_coeff,
                                             step)
                        running_loss = 0.0
                        running_g_loss = 0.0
                        running_q_loss = 0.0

        # save model snapshot
        if args.save_every is not None and epoch % args.save_every == args.save_every - 1:
            torch.save(
                model, '{0}/{1}_{2}_model.pt'.format(save_path, args.run_id,
                                                     epoch + 1))

        # decay learning rate
        if args.lr_decay > 0:
            lr_scheduler.step()
