import os
import json
import argparse
import numpy as np
import torch

import stylegan2
from stylegan2 import utils

#----------------------------------------------------------------------------

_description = """Metrics evaluation.
Run 'python %(prog)s <subcommand> --help' for subcommand help."""

#----------------------------------------------------------------------------

_examples = """examples:
  # Train a network or convert a pretrained one. In this example we first convert a pretrained one.
  python run_convert_from_tf --download ffhq-config-f --output G.pth D.pth Gs.pth

  # Project generated images
  python %(prog)s project_generated_images --network=Gs.pth --seeds=0,1,5

  # Project real images
  python %(prog)s project_real_images --network=Gs.pth --data-dir=path/to/image_folder
"""

#----------------------------------------------------------------------------

def _add_shared_arguments(parser):

    parser.add_argument(
        '--network',
        help='Network file path',
        required=True,
        metavar='FILE'
    )

    parser.add_argument(
        '--num_samples',
        type=int,
        help='Number of samples to gather for evaluating ' + \
            'this metric. Default: %(default)s',
        default=50000,
        metavar='VALUE'
    )

    parser.add_argument(
        '--size',
        type=int,
        help='Rescale images so that this is the size of their ' + \
            'smallest side in pixels. Default: Unscaled',
        default=None,
        metavar='VALUE'
    )

    parser.add_argument(
        '--batch_size',
        help='Batch size for generator. Default: %(default)s',
        type=int,
        default=1,
        metavar='VALUE'
    )

    parser.add_argument(
        '--output',
        help='Root directory for run results. Default: %(default)s',
        type=str,
        default='./results',
        metavar='DIR'
    )

    parser.add_argument(
        '--pixel_min',
        help='Minumum of the value range of pixels in generated images. ' + \
            'Default: %(default)s',
        default=-1,
        type=float,
        metavar='VALUE'
    )

    parser.add_argument(
        '--pixel_max',
        help='Maximum of the value range of pixels in generated images. ' + \
            'Default: %(default)s',
        default=1,
        type=float,
        metavar='VALUE'
    )

    parser.add_argument(
        '--gpu',
        help='CUDA device indices (given as separate ' + \
            'values if multiple, i.e. "--gpu 0 1"). Default: Use CPU',
        type=int,
        default=[],
        nargs='*',
        metavar='INDEX'
    )


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description=_description,
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    fid_parser = subparsers.add_parser('fid', help='Calculate FID')

    fid_parser.add_argument(
        '--data_dir',
        help='Dataset root directory',
        required=True,
        metavar='DIR'
    )

    fid_parser.add_argument(
        '--reals_batch_size',
        help='Batch size for gathering statistics of reals. Default: %(default)s',
        type=int,
        default=1,
        metavar='VALUE'
    )

    fid_parser.add_argument(
        '--reals_data_workers',
        help='Data workers for fetching real data samples. Default: %(default)s',
        type=int,
        default=4,
        metavar='VALUE'
    )

    fid_parser.add_argument(
        '--truncation_psi',
        help='Truncation psi. Default: %(default)s',
        type=float,
        default=1.0,
        metavar='VALUE'
    )

    _add_shared_arguments(fid_parser)

    ppl_parser = subparsers.add_parser('ppl', help='Calculate PPL')

    ppl_parser.add_argument(
        '--epsilon',
        type=float,
        help='Perturbation value. Default: %(default)s',
        default=1e-4,
        metavar='VALUE'
    )

    ppl_parser.add_argument(
        '--use_dlatent',
        type=utils.bool_type,
        help='Measure on perturbations of disentangled latents ' + \
            'instead of raw latents. Default: %(default)s',
        default=True,
        const=True,
        nargs='?',
        metavar='BOOL'
    )

    ppl_parser.add_argument(
        '--full_sampling',
        type=utils.bool_type,
        help='Measure on random interpolation between two inputs ' + \
            'instead of directly on one input. Default: %(default)s',
        default=False,
        const=True,
        nargs='?',
        metavar='BOOL'
    )

    parser.add_argument(
        '--ppl_ffhq_crop',
        help='Crop images evaluated for PPL with crop values ' + \
            'for FFHQ. Default: False',
        type=utils.bool_type,
        const=True,
        nargs='?',
        default=False,
        metavar='BOOL'
    )

    _add_shared_arguments(ppl_parser)

    return parser

#----------------------------------------------------------------------------

def _report_metric(value, name, args):
    fpath = os.path.join(args.output, 'metrics.json')
    metrics = {}
    if os.path.exists(fpath):
        with open(fpath, 'r') as fp:
            try:
                metrics = json.load(fp)
            except Exception:
                pass
    metrics[name] = value
    with open(fpath, 'w') as fp:
        json.dump(metrics, fp)
    print('\n\nMetric evaluated!:')
    print('{}: {}'.format(name, value))

#----------------------------------------------------------------------------

def eval_fid(G, prior_generator, args):
    assert args.data_dir, '--data_dir has to be specified.'
    dataset = utils.ImageFolder(
        args.data_dir,
        pixel_min=args.pixel_min,
        pixel_max=args.pixel_max
    )
    assert len(dataset), 'No images found at {}'.format(args.data_dir)

    inception = stylegan2.external_models.inception.InceptionV3FeatureExtractor(
        pixel_min=args.pixel_min, pixel_max=args.pixel_max)

    if len(args.gpu) > 1:
        inception = torch.nn.DataParallel(inception, device_ids=args.gpu)

    args.reals_batch_size = max(args.reals_batch_size, len(args.gpu))

    fid = stylegan2.metrics.fid.FID(
        G=G,
        prior_generator=prior_generator,
        dataset=dataset,
        num_samples=args.num_samples,
        fid_model=inception,
        fid_size=args.size,
        truncation_psi=args.truncation_psi,
        reals_batch_size=args.reals_batch_size,
        reals_data_workers=args.reals_data_workers
    )

    value = fid.evaluate()

    name = 'FID'
    if args.size:
        name += '({})'.format(args.size)
    if args.truncation_psi != 1:
        name +='trunc{}'.format(args.truncation_psi)
    name += ':{}k'.format(args.num_samples // 1000)

    _report_metric(value, name, args)

#----------------------------------------------------------------------------

def eval_ppl(G, prior_generator, args):

    lpips = stylegan2.external_models.lpips.LPIPS_VGG16(
        pixel_min=args.pixel_min, pixel_max=args.pixel_max)

    if len(args.gpu) > 1:
        lpips = torch.nn.DataParallel(lpips, device_ids=args.gpu)

    crop = None
    if args.ppl_ffhq_crop:
        crop = stylegan2.metrics.ppl.PPL.FFHQ_CROP

    ppl = stylegan2.metrics.ppl.PPL(
        G=G,
        prior_generator=prior_generator,
        num_samples=args.num_samples,
        epsilon=args.epsilon,
        use_dlatent=args.use_dlatent,
        full_sampling=args.full_sampling,
        crop=crop,
        lpips_model=lpips,
        lpips_size=args.size,
    )

    value = ppl.evaluate()

    name = 'PPL'
    if args.size:
        name += '({})'.format(args.size)
    if args.use_dlatent:
        name += 'W'
    else:
        name += 'Z'
    if args.full_sampling:
        name += '-full'
    else:
        name += '-end'
    name += ':{}k'.format(args.num_samples // 1000)

    _report_metric(value, name, args)

#----------------------------------------------------------------------------

def main():
    args = get_arg_parser().parse_args()
    assert args.command, 'Missing subcommand.'
    assert os.path.isdir(args.output) or not os.path.splitext(args.output)[-1], \
        '--output argument should specify a directory, not a file.'
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    G = stylegan2.models.load(args.network)
    assert isinstance(G, stylegan2.models.Generator), 'Model type has to be ' + \
        'stylegan2.models.Generator. Found {}.'.format(type(G))

    latent_size, label_size = G.latent_size, G.label_size

    device = torch.device(args.gpu[0] if args.gpu else 'cpu')
    if device.index is not None:
        torch.cuda.set_device(device.index)

    G.to(device).eval().requires_grad_(False)

    if len(args.gpu) > 1:
        G = torch.nn.DataParallel(G, device_ids=args.gpu)

    args.batch_size = max(args.batch_size, len(args.gpu))

    prior_generator = utils.PriorGenerator(
        latent_size=latent_size,
        label_size=label_size,
        batch_size=args.batch_size,
        device=device
    )

    if args.command == 'fid':
        eval_fid(G, prior_generator, args)
    elif args.command == 'ppl':
        eval_ppl(G, prior_generator, args)
    else:
        raise TypeError('Unkown command {}'.format(args.command))


if __name__ == '__main__':
    main()
