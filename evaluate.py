import argparse
import json
from main import DATASETS, parse_args, project_path
MODEL_SETTINGS = ('hidden', 'embed_size', 'padmaxlen', 'dropout', 'seed', 'state_dim', 'max_parents', 'max_siblings', 'disable_hssga', 'encode_batch_size', 'eval_batch_size')

def evaluation_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--encode_batch_size', type=int)
    parser.add_argument('--eval_batch_size', type=int)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--gpu_id', type=int, default=0)
    options = parser.parse_args(argv)
    checkpoint = project_path(options.checkpoint)
    if not checkpoint.is_file():
        parser.error('Checkpoint missing')
    sidecar = checkpoint.with_name(checkpoint.name + '.metrics.json')
    if not sidecar.is_file():
        parser.error('Checkpoint metadata missing')
    metadata = json.loads(sidecar.read_text(encoding='utf-8'))
    saved = metadata.get('Arguments', {})
    dataset = saved.get('dataset')
    if dataset not in DATASETS:
        parser.error('Checkpoint metadata must include a supported Arguments.dataset value')
    if options.dataset is not None and options.dataset != dataset:
        parser.error('The requested dataset differs from the saved checkpoint dataset')
    arguments = ['--dataset', dataset, '--device', options.device, '--gpu_id', str(options.gpu_id)]
    for name in ('data_dir', 'model_path', 'output_dir'):
        if getattr(options, name):
            arguments.extend(('--' + name, getattr(options, name)))
    args = parse_args(arguments)
    for name in MODEL_SETTINGS:
        if name in saved:
            setattr(args, name, saved[name])
    for name in ('encode_batch_size', 'eval_batch_size'):
        value = getattr(options, name)
        if value is not None:
            if value <= 0:
                parser.error(f'--{name} must be positive')
            setattr(args, name, value)
    return (checkpoint, metadata, args)

def main(argv=None):
    (checkpoint, metadata, args) = evaluation_args(argv)
    import torch
    from transformers import BertTokenizer
    from model import HSSGATaxoBell
    from taxonomy import Taxonomy
    from train import evaluate, write_json
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; use a CUDA environment or --device cpu')
        torch.cuda.set_device(args.gpu_id)
        device = torch.device('cuda', args.gpu_id)
    else:
        device = torch.device('cpu')
    torch.set_float32_matmul_precision('high')
    data = Taxonomy(args.data_dir, args.dataset)
    data.graph.context_seed = args.seed
    tokenizer = BertTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    tokens = tokenizer([name.lower() for name in data.names], text_pair=data.definitions, max_length=args.padmaxlen, padding='max_length', truncation=True, return_tensors='pt')
    model = HSSGATaxoBell(args)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.to(device).eval()
    (metrics, diagnostics, predictions) = evaluate(model, tokens, data, args, device)
    output = args.output_dir / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    stem = 'evaluation_' + checkpoint.stem
    write_json(output / (stem + '.metrics.json'), {'Test Metrics BCE': metrics['BC'], 'Test Metrics Kl': metrics['KL'], 'Diagnostics': diagnostics})
    with (output / (stem + '.predictions.jsonl')).open('w', encoding='utf-8') as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'dataset': args.dataset, 'checkpoint': checkpoint.name, 'metrics': metrics}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
