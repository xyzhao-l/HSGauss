import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from main import serializable_args
from metrics import RankingMetrics
from model import HSSGATaxoBell
from taxonomy import Taxonomy, build_batch, build_context_batch

def write_json(path, content):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)

def save_weights(path, model):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, path)

@torch.no_grad()
def project_candidates(model, features, data, args, excluded_query=-1):
    (states, information) = ([], [])
    for start in range(0, len(data.candidates), args.eval_batch_size):
        pairs = [(excluded_query, node) for node in data.candidates[start:start + args.eval_batch_size]]
        layout = build_context_batch(data, pairs, args.max_parents, args.max_siblings, enabled=not args.disable_hssga)
        (box, details) = model.candidate_boxes(features[layout['node_ids']], layout)
        states.append(model.box_state(box))
        information.append(details)
    state = {key: torch.cat([part[key] for part in states]) for key in states[0]}
    details = {key: torch.cat([part[key] for part in information]) for key in information[0]}
    if not all((torch.isfinite(value).all() for value in state.values())):
        raise FloatingPointError('Non-finite candidate Gaussian parameter')
    return (state, details)

def context_report(details):
    if not details:
        return {'context_coverage': 0.0}
    available = details['context_available']
    report = {'context_coverage': available.float().mean().item()}
    if not available.any():
        return report
    for key in ('precision_ratio', 'effective_weight', 'std_ratio'):
        values = details[key][available]
        report[key + '_min'] = values.min().item()
        report[key + '_max'] = values.max().item()
    shifts = details['standardized_mean_shift'][available].abs()
    report['mean_abs_shift'] = shifts.mean().item()
    report['max_abs_shift'] = shifts.max().item()
    for key in ('context_norm', 'retention'):
        report[key + '_mean'] = details[key][available].mean().item()
    return report

@torch.no_grad()
def evaluate(model, tokens, data, args, device, features=None):
    model.eval()
    if features is None:
        features = model.encode(tokens, list(range(len(data.names))), device, args.encode_batch_size)
    query_states = model.text_state(features[data.queries])
    if not all((torch.isfinite(value).all() for value in query_states.values())):
        raise FloatingPointError('Non-finite query Gaussian parameter')
    common = None
    context_details = []
    accumulators = {name: RankingMetrics(data.candidates, data.multi_parent, data.paths) for name in ('BC', 'KL')}
    ranking = {name: {'root_top1': 0, 'ancestor_overreach_errors': 0} for name in accumulators}
    roots = {node for node in data.graph.nodes if not data.graph.parents[node]}
    predictions = []
    for (query_index, (query_id, expected)) in enumerate(zip(data.queries, data.ground_truth)):
        query = {key: value[query_index:query_index + 1] for (key, value) in query_states.items()}
        if query_id in data.graph.nodes and (not args.disable_hssga):
            (candidates, details) = project_candidates(model, features, data, args, query_id)
        else:
            if common is None:
                common = project_candidates(model, features, data, args)
            (candidates, details) = common
        context_details.append(details)
        values = {name: [] for name in accumulators}
        for start in range(0, len(data.candidates), args.eval_batch_size):
            scores = model.scores(query, {key: value[start:start + args.eval_batch_size] for (key, value) in candidates.items()})
            for (name, score) in scores.items():
                values[name].append(score)
        values = {name: torch.cat(parts) for (name, parts) in values.items()}
        if not all((torch.isfinite(value).all() for value in values.values())):
            raise FloatingPointError('Non-finite ranking score')
        record = {'query': data.names[query_id], 'parents': [data.names[node] for node in expected]}
        ancestors = {node for parent in expected for (node, _) in data.graph.ancestors(parent, hops=len(data.graph.nodes), excluded={query_id})} - set(expected)
        for (name, accumulator) in accumulators.items():
            indices = values[name].argsort(descending=True, stable=True).cpu().tolist()
            accumulator.update(indices, expected)
            predicted = data.candidates[indices[0]]
            ranking[name]['root_top1'] += int(predicted in roots)
            ranking[name]['ancestor_overreach_errors'] += int(predicted in ancestors)
            scores = values[name].cpu().tolist()
            record[name] = [{'concept': data.names[data.candidates[index]], 'score': scores[index]} for index in indices[:10]]
        predictions.append(record)
    details = {key: torch.cat([part[key] for part in context_details]) for key in context_details[0]}
    return ({name: metric.result() for (name, metric) in accumulators.items()}, {'hssga': context_report(details), 'ranking': ranking}, predictions)

def train(args):
    if not (args.data_dir / args.dataset).is_dir():
        raise FileNotFoundError('Dataset directory missing')
    for filename in ('config.json', 'vocab.txt'):
        if not (args.model_path / filename).is_file():
            raise FileNotFoundError(f'BERT file missing: {filename}')
    if not any(((args.model_path / filename).is_file() for filename in ('model.safetensors', 'pytorch_model.bin'))):
        raise FileNotFoundError('BERT weights missing')
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; check the server environment')
        torch.cuda.set_device(args.gpu_id)
        device = torch.device('cuda', args.gpu_id)
    else:
        device = torch.device('cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision('high')
    data = Taxonomy(args.data_dir, args.dataset)
    data.graph.context_seed = args.seed
    triples = data.training_triples(args.negsamples, args.seed, args.expID)
    cf_enabled = not args.disable_hssga and args.cf_drop_rate > 0 and (args.cf_bc_weight > 0 or args.cf_kl_weight > 0)
    report = {'concepts': len(data.names), 'train_edges': len(data.train_edges), 'train_triples': len(triples), 'queries': len(data.queries), 'candidates': len(data.candidates), 'skipped_queries': data.skipped_queries, 'queries_in_seed_graph': len(data.overlap_queries), 'cached_negative_conflicts': data.cached_negative_conflicts}
    tokenizer = BertTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    tokens = tokenizer([name.lower() for name in data.names], text_pair=data.definitions, max_length=args.padmaxlen, padding='max_length', truncation=True, return_tensors='pt')
    model = HSSGATaxoBell(args).to(device)
    groups = [{'params': list(model.bert.parameters()), 'lr': args.lr}, {'params': [parameter for (name, parameter) in model.named_parameters() if not name.startswith('pre_train_model.')], 'lr': args.lr_proj}]
    optimizer = torch.optim.AdamW(groups, eps=1e-08, weight_decay=0.0)
    cf_rng = random.Random(args.seed)
    loader = DataLoader(triples, batch_size=args.batch_size, shuffle=True, collate_fn=list, num_workers=0)
    result_dir = args.output_dir / args.dataset
    result_dir.mkdir(parents=True, exist_ok=True)
    suffix = '_baseline' if args.disable_hssga else ''
    arguments = serializable_args(args)
    write_json(result_dir / f'arguments{suffix}.json', {'Arguments': arguments, 'Data': report})
    log_path = result_dir / f'train{suffix}.jsonl'
    log_path.write_text('', encoding='utf-8')
    best_mrr = {name: -math.inf for name in ('BC', 'KL')}
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        (totals, examples) = ({}, 0)
        for batch_triples in tqdm(loader, desc=f'Epoch {epoch:03d}/{args.epochs}', leave=False):
            batch = build_batch(data, batch_triples, args.max_parents, args.max_siblings, enabled=not args.disable_hssga, cf_drop_rate=args.cf_drop_rate if cf_enabled else 0.0, cf_rng=cf_rng)
            optimizer.zero_grad(set_to_none=True)
            (loss, components) = model(tokens, batch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch}')
            loss.backward()
            optimizer.step()
            count = len(batch_triples)
            examples += count
            for (name, value) in components.items():
                totals[name] = totals.get(name, 0.0) + value.item() * count
        losses = {name: value / examples for (name, value) in totals.items()}
        (metrics, diagnostics, predictions) = evaluate(model, tokens, data, args, device)
        entry = {'epoch': epoch, 'train_loss': losses['loss'], 'Training Statistics': losses, 'Test Metrics BCE': metrics['BC'], 'Test Metrics Kl': metrics['KL'], 'Diagnostics': diagnostics, 'epoch_seconds': time.monotonic() - epoch_started}
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
        for name in best_mrr:
            if metrics[name]['MRR'] >= best_mrr[name]:
                best_mrr[name] = metrics[name]['MRR']
                checkpoint = result_dir / f'{name}{suffix}.pt'
                save_weights(checkpoint, model)
                write_json(checkpoint.with_name(checkpoint.name + '.metrics.json'), {'Arguments': arguments, 'Data': report, **entry})
                with (result_dir / f'{name}_predictions{suffix}.jsonl').open('w', encoding='utf-8') as handle:
                    for prediction in predictions:
                        handle.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + '\n')
        save_weights(result_dir / f'latest{suffix}.pt', model)
        write_json(result_dir / f'latest{suffix}.pt.metrics.json', {'Arguments': arguments, **entry})
        score_text = ' '.join((f"{name}: MRR={value['MRR']:.5f} R@1={value['Recall@1']:.5f}" for (name, value) in metrics.items()))
        print(f"Epoch {epoch:03d} loss={losses['loss']:.5f} {score_text} bc_keep={losses['cf_bc_keep_loss']:.4f} kl_rob={losses['cf_kl_robust_loss']:.4f} cf_active={losses['cf_active_fraction']:.3f} context={diagnostics['hssga']['context_coverage']:.3f} shift={diagnostics['hssga'].get('mean_abs_shift', 0.0):.4f} time={entry['epoch_seconds']:.1f}s", flush=True)
    print(f'Finished {args.dataset}: {time.monotonic() - started:.1f}s; best MRR {best_mrr}', flush=True)
