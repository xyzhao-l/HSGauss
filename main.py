import argparse
from pathlib import Path
ROOT = Path(__file__).absolute().parent
DATASETS = ('environment', 'science', 'wordnet', 'semeval_food', 'mesh')

def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path

def serializable_args(args):
    return {key: '' if isinstance(value, Path) or key.endswith(('_dir', '_path')) or key in ('path', 'checkpoint', 'triple_cache') else value for (key, value) in vars(args).items()}

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, default='environment')
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--epochs', type=int, default=125)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--encode_batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--lr', type=float, default=9e-05)
    parser.add_argument('--lr_proj', type=float, default=0.001)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--embed_size', type=int, default=8)
    parser.add_argument('--padmaxlen', type=int, default=30)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--negsamples', type=int, default=None)
    parser.add_argument('--expID', type=int, default=0)
    parser.add_argument('--wtbce', type=float, default=0.45)
    parser.add_argument('--wtkl', type=float, default=0.45)
    parser.add_argument('--wtreg', type=float, default=0.1)
    parser.add_argument('--lam', type=float, default=0.3)
    parser.add_argument('--C', type=float, default=1.5)
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--min_variance', type=float, default=0.25)
    parser.add_argument('--max_variance', type=float, default=10.0)
    parser.add_argument('--state_dim', type=int, default=64)
    parser.add_argument('--max_parents', type=int, default=4)
    parser.add_argument('--max_siblings', type=int, default=4)
    parser.add_argument('--cf_bc_weight', type=float, default=0.001)
    parser.add_argument('--cf_kl_weight', type=float, default=0.01)
    parser.add_argument('--bc_keep_ratio', type=float, default=0.8)
    parser.add_argument('--cf_drop_rate', type=float, default=0.25)
    parser.add_argument('--disable_hssga', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args(argv)
    if args.negsamples is None:
        args.negsamples = {'wordnet': 10, 'mesh': 20, 'semeval_food': 20}.get(args.dataset, 50)
    for name in ('epochs', 'batch_size', 'encode_batch_size', 'eval_batch_size', 'hidden', 'embed_size', 'padmaxlen', 'negsamples', 'state_dim', 'max_parents', 'max_siblings'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    if args.gpu_id < 0 or not 0 <= args.dropout < 1:
        parser.error('gpu_id must be nonnegative and dropout must be in [0, 1)')
    if not 0 < args.min_variance < args.max_variance:
        parser.error('Require 0 < min_variance < max_variance')
    if min(args.lr, args.lr_proj, args.margin) <= 0:
        parser.error('Learning rates and margin must be positive')
    if min(args.wtbce, args.wtkl, args.wtreg, args.lam, args.C) < 0:
        parser.error('Loss coefficients must be nonnegative')
    if any((not 0 <= weight < float('inf') for weight in (args.cf_bc_weight, args.cf_kl_weight))):
        parser.error('Counterfactual loss weights must be finite and nonnegative')
    if not 0 <= args.cf_drop_rate < 1 or not 0 < args.bc_keep_ratio <= 1:
        parser.error('cf_drop_rate must be in [0, 1); bc_keep_ratio must be in (0, 1]')
    for name in ('data_dir', 'model_path', 'output_dir'):
        value = getattr(args, name)
        if not value or not str(value).strip():
            parser.error(f'--{name} required')
        setattr(args, name, project_path(value))
    args.model = 'bert'
    args.method = 'normal'
    args.state_hops = 2
    args.fusion_rho = 0.1
    args.context_mean_bound = 0.5
    args.context_precision_ratio_bound = 2.0
    return args
if __name__ == '__main__':
    arguments = parse_args()
    from train import train
    train(arguments)
