#!/usr/bin/env python
# coding: utf-8

import os
import os.path
import json
import argparse
from random import seed
import re
import time as time_module

try:
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer
except ImportError:
    stopwords = None
    WordNetLemmatizer = None

from sklearn.feature_extraction.text import TfidfVectorizer

import torch

from torch.utils.data import DataLoader

from collections import defaultdict
from encoders import *
from quantizers import *
from train import *
from utils.data import *
from utils.loss import *
from utils.summary import truncate_summary, RougeEvaluator


# parts of the code has been
# adapted from: https://github.com/stangelid/qt


def load_semae_model(model_path, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Extracts aspect summaries with a trained SemAE model.\n')

    data_arg_group = argparser.add_argument_group('Data arguments')
    data_arg_group.add_argument('--summary_data',
                                help='summarization benchmark data',
                                type=str,
                                default='../data/space/json/space_summ.json')
    data_arg_group.add_argument('--gold_data',
                                help='gold data root directory',
                                type=str,
                                default='../data/space/gold')
    data_arg_group.add_argument(
        '--gold_aspects',
        help=
        'aspect categories to evaluate against (default: all SPACE aspects)',
        type=str,
        default='building,cleanliness,food,location,rooms,service')
    argparser.add_argument(
        '--seedsdir',
        help='directory that holds aspect query words, i.e., seeds',
        type=str,
        default='../data/seeds')
    argparser.add_argument('--max_num_seeds',
                           help='number of seed words per aspect',
                           type=int,
                           default=5)
    data_arg_group.add_argument(
        '--sentencepiece',
        help='sentencepiece model file',
        type=str,
        default='../data/sentencepiece/spm_unigram_32k.model')
    data_arg_group.add_argument(
        '--max_num_entities',
        help='maximum number of entities to load for summarization (default: all)',
        type=int,
        default=None)
    data_arg_group.add_argument(
        '--max_rev_len',
        help='maximum number of sentences per review (default: 150)',
        type=int,
        default=150)
    data_arg_group.add_argument(
        '--max_sen_len',
        help='maximum number of tokens per sentence (default: 40)',
        type=int,
        default=40)
    data_arg_group.add_argument(
        '--split_by',
        help='how to split summary data (use "alphanum" for SPACE)',
        type=str,
        default='alphanum')

    summ_arg_group = argparser.add_argument_group('Summarizer arguments')
    summ_arg_group.add_argument('--model',
                                help='trained QT model to use',
                                type=str,
                                default='../models/run11_20_model.pt')
    summ_arg_group.add_argument(
        '--manual_head',
        help=
        'manually set aspect head for extraction (default: auto via entropy)',
        type=int,
        default=None)
    summ_arg_group.add_argument(
        '--truncate_clusters',
        help=
        'truncate cluster sampling to top-p % of clusters (if < 1) or top-k (if > 1)',
        type=float,
        default=1.0)
    summ_arg_group.add_argument('--beta',
                                help='Parameter beta',
                                type=float,
                                default=0.7)
    summ_arg_group.add_argument(
        '--num_cluster_samples',
        help='number of cluster samples (default: 300)',
        type=int,
        default=300)
    summ_arg_group.add_argument(
        '--sample_sentences',
        help=
        'enable 2-step sampling (sample sentences within cluster neighbourhood)',
        action='store_true')
    summ_arg_group.add_argument(
        '--truncate_cluster_nn',
        help=
        'truncate sentences that live in a cluster neighborhood (default: 5)',
        type=int,
        default=5)
    summ_arg_group.add_argument(
        '--num_sent_samples',
        help='number of sentence samples per cluster sample (default: 30)',
        type=int,
        default=30)
    summ_arg_group.add_argument(
        '--temp',
        help='temperature for sampling sentences within cluster (default: 10)',
        type=int,
        default=3)

    out_arg_group = argparser.add_argument_group('Output control')
    out_arg_group.add_argument('--outdir',
                               help='directory to put summaries',
                               type=str,
                               default='../outputs')
    out_arg_group.add_argument('--max_tokens',
                               help='summary budget in words (default: 40)',
                               type=int,
                               default=40)
    out_arg_group.add_argument(
        '--min_tokens',
        help='minimum summary sentence length in words (default: 1)',
        type=int,
        default=1)
    out_arg_group.add_argument(
        '--cos_thres',
        help='cosine similarity threshold for extraction (default: 1.0)',
        type=float,
        default=1.0)
    out_arg_group.add_argument('--no_cut_sents',
                               help='don\'t cut last summary sentence',
                               action='store_true')
    out_arg_group.add_argument('--no_early_stop',
                               help='allow last sentence to go over limit',
                               action='store_true')
    out_arg_group.add_argument(
        '--newline_sentence_split',
        help='one sentence per line (don\'t use if evaluating with ROUGE)',
        action='store_true')
    out_arg_group.add_argument(
        '--sentiment_split',
        help='also write per-sentiment files in <output_path>_sentiment/',
        action='store_true')
    out_arg_group.add_argument(
        '--taxonomy',
        help='taxonomy JSON for sentiment keywords (default: ../data/hasos/aspect_taxonomy.json)',
        type=str,
        default='../data/hasos/aspect_taxonomy.json')
    out_arg_group.add_argument(
        '--trace_jsonl',
        help='optional JSONL trace path for pipeline stage/sample logs',
        type=str,
        default='')
    out_arg_group.add_argument(
        '--trace_sample_limit',
        help='max ranked/write sample rows to emit into trace_jsonl',
        type=int,
        default=30)

    other_arg_group = argparser.add_argument_group('Other arguments')
    other_arg_group.add_argument('--run_id',
                                 help='unique run id (for outputs)',
                                 type=str,
                                 default='aspect_run_new')
    other_arg_group.add_argument('--no_eval',
                                 help='don\'t evaluate (just write summaries)',
                                 action='store_true')
    other_arg_group.add_argument(
        '--gpu',
        help='gpu device to use (default: -1, i.e., use cpu)',
        type=int,
        default=1)
    other_arg_group.add_argument('--batch_size',
                                 help='the maximum batch size (default: 5)',
                                 type=int,
                                 default=5)
    other_arg_group.add_argument('--sfp',
                                 help='system filename pattern for pyrouge',
                                 type=str,
                                 default='(.*)')
    other_arg_group.add_argument('--mfp',
                                 help='model filename pattern for pyrouge',
                                 type=str,
                                 default='#ID#_[012].txt')
    other_arg_group.add_argument('--seed',
                                 help='random seed',
                                 type=int,
                                 default=1)
    other_arg_group.add_argument('--shard_idx',
                                 help='which shard to run (0-based)',
                                 type=int,
                                 default=0)
    other_arg_group.add_argument('--num_shards',
                                 help='total number of shards for entity-level parallelism',
                                 type=int,
                                 default=1)
    args = argparser.parse_args()

    seed(1)

    if args.gpu >= 0:
        device = torch.device('cuda:{0}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    # set paths
    summ_data_path = args.summary_data
    model_path = args.model
    output_path = os.path.join(args.outdir, args.run_id)
    eval_path = args.outdir
    gold_path = args.gold_data
    seeds_path = args.seedsdir
    spm_path = args.sentencepiece
    beta = args.beta

    assert args.model != '', 'Please give model path'

    trace_file = None

    def emit_trace(stage, **payload):
        if trace_file is None:
            return
        row = {
            'time': time_module.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'stage': stage,
            'run_id': args.run_id,
            'shard_idx': args.shard_idx,
            'num_shards': args.num_shards,
        }
        row.update(payload)
        trace_file.write(json.dumps(row, ensure_ascii=False) + '\n')
        trace_file.flush()

    if args.trace_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.trace_jsonl)),
                  exist_ok=True)
        trace_file = open(args.trace_jsonl, 'w', encoding='utf-8')
        emit_trace('start',
                   summary_data=summ_data_path,
                   model=model_path,
                   sentencepiece=spm_path,
                   seedsdir=seeds_path,
                   output_path=output_path,
                   sentiment_split=bool(args.sentiment_split),
                   max_tokens=args.max_tokens,
                   beta=beta)

    # read aspect seed words
    aspects = args.gold_aspects.split(',')
    num_aspects = len(aspects)
    aspect_indices = {}
    aspect_seeds = {}

    for i, aspect in enumerate(aspects):
        aspect_indices[aspect] = i
        seeds = {}
        f = open(os.path.join(seeds_path, aspect + '.txt'), 'r')
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                seed_word = parts[0]
            else:
                seed_word = parts[1]
            seeds[seed_word.lower()] = True
        f.close()
        aspect_seeds[aspect] = seeds
    emit_trace('seeds_loaded',
               aspects=len(aspects),
               max_num_seeds=args.max_num_seeds,
               seedsdir=seeds_path)

    # aspect mapping tools
    token_pattern = re.compile(r'(?u)\b\w\w+\b')
    try:
        if stopwords is None:
            raise LookupError
        stop_words = set(stopwords.words('english'))
    except LookupError:
        stop_words = set(TfidfVectorizer(stop_words='english').get_stop_words())
    if WordNetLemmatizer is None:
        lemmatizer = None
    else:
        lemmatizer = WordNetLemmatizer()
        try:
            lemmatizer.lemmatize('rooms')
        except LookupError:
            lemmatizer = None

    # load summarization data
    f = open(summ_data_path, 'r', encoding='utf-8')
    summ_data = json.load(f)
    f.close()
    emit_trace('summary_data_loaded',
               entities=len(summ_data) if hasattr(summ_data, '__len__') else None)

    # shard slicing for parallel runs (round-robin so review-counts are balanced)
    if args.num_shards > 1:
        if isinstance(summ_data, list):
            summ_data = summ_data[args.shard_idx::args.num_shards]
        else:
            keys = sorted(summ_data.keys())[args.shard_idx::args.num_shards]
            summ_data = {k: summ_data[k] for k in keys}
        print('[shard {0}/{1}] processing {2} entities'.format(
            args.shard_idx, args.num_shards, len(summ_data)), flush=True)
        emit_trace('shard_selected',
                   entities=len(summ_data),
                   shard_idx=args.shard_idx,
                   num_shards=args.num_shards)

    # prepare summarization dataset
    summ_dataset = ReviewSummarizationDataset(summ_data,
                                              sample_size=args.max_num_entities,
                                              spmodel=spm_path,
                                              max_rev_len=args.max_rev_len,
                                              max_sen_len=args.max_sen_len)
    vocab_size = summ_dataset.vocab_size
    pad_id = summ_dataset.pad_id()
    bos_id = summ_dataset.bos_id()
    eos_id = summ_dataset.eos_id()
    unk_id = summ_dataset.unk_id()
    emit_trace('dataset_prepared',
               vocab_size=vocab_size,
               pad_id=pad_id,
               bos_id=bos_id,
               eos_id=eos_id,
               unk_id=unk_id)

    # wrapper for collate function
    collator = ReviewCollator(padding_idx=pad_id,
                              unk_idx=unk_id,
                              bos_idx=bos_id,
                              eos_idx=eos_id)

    # split dev/test entities
    summ_dataset.entity_split(split_by=args.split_by)

    # create entity data loaders
    summ_dls = {}
    summ_samplers = summ_dataset.get_entity_batch_samplers(args.batch_size)
    for entity_id, entity_sampler in summ_samplers.items():
        summ_dls[entity_id] = DataLoader(
            summ_dataset,
            batch_sampler=entity_sampler,
            collate_fn=collator.collate_reviews_with_ids)

    torch.manual_seed(args.seed)

    # Model Loading

    model = load_semae_model(args.model, device)
    model.to(device)
    nheads = model.encoder.output_nheads
    codebook_size = model.codebook_size
    d_model = model.d_model
    model.eval()
    emit_trace('model_loaded',
               nheads=nheads,
               codebook_size=codebook_size,
               d_model=d_model,
               device=str(device))

    # Prepare Aspect data
    def normalize_tokens(text):
        tokens = [
            tok for tok in token_pattern.findall(text.lower())
            if tok not in stop_words
        ]
        if lemmatizer is not None:
            tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
        return tokens

    def seed_matches(seed_term, token_set, normalized_text):
        seed_tokens = normalize_tokens(seed_term)
        if not seed_tokens:
            return False
        if len(seed_tokens) == 1:
            return seed_tokens[0] in token_set
        return ' '.join(seed_tokens) in normalized_text

    def get_aspects(sent):
        sent_tokens = normalize_tokens(sent)
        sent_token_set = set(sent_tokens)
        normalized_text = ' '.join(sent_tokens)
        results = []
        for aspect in aspects:
            for x in aspect_seeds[aspect].keys():
                if seed_matches(x, sent_token_set, normalized_text):
                    results.append(aspect)
                    break

        if not results:
            return ["others"]
        return results

    all_texts = []

    # Load taxonomy sentiment keywords for --sentiment_split
    taxonomy_sentiment = {}
    if args.sentiment_split:
        with open(args.taxonomy, 'r', encoding='utf-8') as tf:
            taxonomy_data = json.load(tf)
        # Handle both formats: {"aspects": [...]} or [...]
        entries = taxonomy_data['aspects'] if isinstance(taxonomy_data, dict) else taxonomy_data
        for entry in entries:
            code = entry.get('CODE') or entry.get('code')
            pos_key = 'POSITIVE_SENTIMENT_KEYWORDS' if 'POSITIVE_SENTIMENT_KEYWORDS' in entry else 'positive_sentiment_keywords'
            neg_key = 'NEGATIVE_SENTIMENT_KEYWORDS' if 'NEGATIVE_SENTIMENT_KEYWORDS' in entry else 'negative_sentiment_keywords'
            neu_key = 'NEUTRAL_SENTIMENT_KEYWORDS' if 'NEUTRAL_SENTIMENT_KEYWORDS' in entry else 'neutral_sentiment_keywords'
            taxonomy_sentiment[code] = {
                'pos': set(w.lower() for w in entry.get(pos_key, [])),
                'neg': set(w.lower() for w in entry.get(neg_key, [])),
                'neu': set(w.lower() for w in entry.get(neu_key, [])),
            }
        print('[sentiment] loaded keywords for {0} aspects'.format(len(taxonomy_sentiment)))
        emit_trace('sentiment_taxonomy_loaded',
                   aspects=len(taxonomy_sentiment),
                   taxonomy=args.taxonomy,
                   method='taxonomy_keyword_post_rank')

    def classify_sentiment(sentence, aspect):
        """Keyword-based sentiment: count keyword hits, return pos/neg/neu."""
        if aspect not in taxonomy_sentiment:
            return 'neu'
        sent_lower = sentence.lower()
        scores = {}
        for label, keywords in taxonomy_sentiment[aspect].items():
            scores[label] = sum(1 for kw in keywords if kw in sent_lower)
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return 'neu'
        return best

    ranked_entity_sentences = defaultdict(dict)
    ranked_entity_records = defaultdict(dict)
    trace_ranked_samples = 0

    with torch.no_grad():
        for entity_id, entity_loader in tqdm(summ_dls.items()):
            current_entity_id = entity_id
            texts = []
            text_sources = []
            distances = []

            for batch in entity_loader:
                src = batch[0].to(device)
                ids = batch[2]
                for full_id in ids:
                    source_entity_id, review_id = full_id.split('__')
                    review_sents = summ_dataset.reviews[source_entity_id][
                        review_id][:args.max_rev_len]
                    texts.extend(review_sents)
                    text_sources.extend([{
                        'entity_id': source_entity_id,
                        'review_id': review_id
                    } for _ in review_sents])

                batch_size, nsent, ntokens = src.size()

                _, _, _, dist = model.cluster(src)
                distances.extend(dist)

            aspect_wise_sentences = defaultdict(list)
            for i, sentence in enumerate(texts):
                sent_aspects = list(get_aspects(sentence))

                if len(sent_aspects) == 0:
                    continue

                for aspect in sent_aspects:
                    aspect_wise_sentences[aspect].append(i)

            distances = torch.stack(distances)
            P_k = torch.mean(distances, dim=0)
            emit_trace('entity_encoded',
                       entity_id=current_entity_id,
                       sentences=len(texts),
                       aspect_candidates=len(aspect_wise_sentences))

            for aspect in aspect_wise_sentences.keys():
                aspect_dist = []
                for idx in aspect_wise_sentences[aspect]:
                    aspect_dist.append(distances[idx])

                aspect_dist = torch.stack(aspect_dist)
                P_z = torch.mean(aspect_dist, dim=0)

                # Form ranked kl divergence list
                kl_divs = []
                for i in range(distances.shape[0]):
                    D_z = distances[i]
                    kl_divs.append(
                        kl_div_all_heads(D_z, P_z) -
                        beta * kl_div_all_heads(D_z, P_k))

                dist = torch.stack(kl_divs).detach().cpu().numpy()
                ranked_sentence_indices = np.argsort(dist)

                ranked_sentence_texts = [
                    texts[idx] for idx in ranked_sentence_indices
                ]
                ranked_entity_sentences[current_entity_id][
                    aspect] = ranked_sentence_texts
                ranked_records = []
                for rank, idx in enumerate(ranked_sentence_indices, 1):
                    source = text_sources[idx] if idx < len(text_sources) else {}
                    ranked_records.append({
                        'rank': rank,
                        'score': float(dist[idx]),
                        'sentence': texts[idx],
                        'review_id': source.get('review_id'),
                    })
                ranked_entity_records[current_entity_id][aspect] = ranked_records
                if trace_ranked_samples < args.trace_sample_limit:
                    for rec in ranked_records[:3]:
                        if trace_ranked_samples >= args.trace_sample_limit:
                            break
                        emit_trace('ranked_sentence_sample',
                                   entity_id=current_entity_id,
                                   aspect=aspect,
                                   rank=rec['rank'],
                                   score=rec['score'],
                                   review_id=rec.get('review_id'),
                                   sentence=rec['sentence'])
                        trace_ranked_samples += 1

            all_texts.extend(texts)

    # tfidf vectorizer used for cosine threshold
    if args.cos_thres != -1:
        vectorizer = TfidfVectorizer(decode_error='replace',
                                     stop_words='english')
        vectorizer.fit(all_texts)
    else:
        vectorizer = None

    # write summaries
    dict_results = {'dev': {}, 'test': {}, 'all': {}}
    all_outputs = []
    trace_write_samples = 0

    if args.newline_sentence_split:
        delim = '\n'
    else:
        delim = '\t'

    for aspect in tqdm(aspects):
        aspect_output_path = os.path.join(output_path, aspect)
        os.makedirs(aspect_output_path, exist_ok=True)

        for entity_id in ranked_entity_sentences:
            if entity_id in summ_dataset.dev_entity_ids:
                file_path = os.path.join(aspect_output_path,
                                         'dev_' + entity_id)
            else:
                file_path = os.path.join(aspect_output_path,
                                         'test_' + entity_id)

            ranked_sentences = ranked_entity_sentences[entity_id].get(aspect, [])
            if ranked_sentences:
                summary_sentences = truncate_summary(
                    ranked_sentences,
                    max_tokens=args.max_tokens,
                    cut_sents=(not args.no_cut_sents),
                    vectorizer=vectorizer,
                    cosine_threshold=args.cos_thres,
                    early_stop=(not args.no_early_stop),
                    min_tokens=args.min_tokens)
            else:
                summary_sentences = []

            fout = open(file_path, 'w', encoding='utf-8')
            fout.write(delim.join(summary_sentences))
            fout.close()
            if trace_write_samples < args.trace_sample_limit:
                emit_trace('summary_written',
                           entity_id=entity_id,
                           aspect=aspect,
                           output_path=file_path,
                           sentences=len(summary_sentences),
                           summary=delim.join(summary_sentences))
                trace_write_samples += 1

            # Sentiment-split write
            if args.sentiment_split:
                sentiment_root = output_path + '_sentiment'
                buckets = {'pos': [], 'neg': [], 'neu': []}
                for sent in summary_sentences:
                    label = classify_sentiment(sent, aspect)
                    buckets[label].append(sent)

                prefix = 'dev_' if entity_id in summ_dataset.dev_entity_ids else 'test_'
                for label, sents in buckets.items():
                    sent_dir = os.path.join(sentiment_root,
                                            '{0}__{1}'.format(aspect, label))
                    os.makedirs(sent_dir, exist_ok=True)
                    sent_path = os.path.join(sent_dir, prefix + entity_id)
                    with open(sent_path, 'w', encoding='utf-8') as fs:
                        fs.write(delim.join(sents))
                    if trace_write_samples < args.trace_sample_limit:
                        emit_trace('sentiment_written',
                                   entity_id=entity_id,
                                   aspect=aspect,
                                   sentiment=label,
                                   output_path=sent_path,
                                   sentences=len(sents))
                        trace_write_samples += 1

        if args.no_eval:
            continue

        # evaluate summaries
        model_dir = os.path.join(gold_path, aspect)
        dev_evaluator = RougeEvaluator(system_dir=aspect_output_path,
                                       model_dir=model_dir,
                                       system_filename_pattern='dev_' +
                                       args.sfp,
                                       model_filename_pattern=args.mfp)
        test_evaluator = RougeEvaluator(system_dir=aspect_output_path,
                                        model_dir=model_dir,
                                        system_filename_pattern='test_' +
                                        args.sfp,
                                        model_filename_pattern=args.mfp)
        all_evaluator = RougeEvaluator(system_dir=aspect_output_path,
                                       model_dir=model_dir,
                                       system_filename_pattern='[^_]*_' +
                                       args.sfp,
                                       model_filename_pattern=args.mfp)

        outputs = dev_evaluator.evaluate()
        dict_results['dev'][aspect] = outputs['dict_output']
        all_outputs.append('{0} vs {1} [dev]'.format(args.run_id, aspect))
        all_outputs.append(outputs['short_output'] + '\n')

        outputs = test_evaluator.evaluate()
        dict_results['test'][aspect] = outputs['dict_output']
        all_outputs.append('{0} vs {1} [test]'.format(args.run_id, aspect))
        all_outputs.append(outputs['short_output'] + '\n')

        outputs = all_evaluator.evaluate()
        dict_results['all'][aspect] = outputs['dict_output']
        all_outputs.append('{0} vs {1} [all]'.format(args.run_id, aspect))
        all_outputs.append(outputs['short_output'] + '\n')

    if args.no_eval:
        print('Skipping ROUGE evaluation because --no_eval was set.')
        emit_trace('complete', no_eval=True)
        if trace_file is not None:
            trace_file.close()
        raise SystemExit(0)

    ftxt = open(os.path.join(eval_path, 'eval_{0}.txt'.format(args.run_id)),
                'w')
    ftxt.write('\n'.join(all_outputs))
    ftxt.close()

    fjson = open(os.path.join(eval_path, 'eval_{0}.json'.format(args.run_id)),
                 'w',
                 encoding='utf-8')
    fjson.write(json.dumps(dict_results))
    fjson.close()
    emit_trace('complete', no_eval=False)
    if trace_file is not None:
        trace_file.close()
