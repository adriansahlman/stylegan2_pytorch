import os
import argparse
import numpy as np
import torch

import stylegan2
from stylegan2 import utils

#----------------------------------------------------------------------------

_description = """StyleGAN2 projector.
Run 'python %(prog)s <subcommand> --help' for subcommand help."""

#----------------------------------------------------------------------------

_examples = """examples:
  # Train a network or convert a pretrained one.
  # Example of converting pretrained ffhq model:
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
        '--num_steps',
        type=int,
        help='Number of steps to use for projection. ' + \
            'Default: %(default)s',
        default=1000,
        metavar='VALUE'
    )

    parser.add_argument(
        '--batch_size',
        help='Batch size. Default: %(default)s',
        type=int,
        default=1,
        metavar='VALUE'
    )

    parser.add_argument(
        '--label',
        help='Label to use for dlatent statistics gathering ' + \
            '(should be integer index of class). Default: no label.',
        type=int,
        default=None,
        metavar='CLASS_INDEX'
    )

    parser.add_argument(
        '--initial_learning_rate',
        help='Initial learning rate of projection. Default: %(default)s',
        default=0.1,
        type=float,
        metavar='VALUE'
    )

    parser.add_argument(
        '--initial_noise_factor',
        help='Initial noise factor of projection. Default: %(default)s',
        default=0.05,
        type=float,
        metavar='VALUE'
    )

    parser.add_argument(
        '--lr_rampdown_length',
        help='Learning rate rampdown length for projection. ' + \
            'Should be in range [0, 1]. Default: %(default)s',
        default=0.25,
        type=float,
        metavar='VALUE'
    )

    parser.add_argument(
        '--lr_rampup_length',
        help='Learning rate rampup length for projection. ' + \
            'Should be in range [0, 1]. Default: %(default)s',
        default=0.05,
        type=float,
        metavar='VALUE'
    )

    parser.add_argument(
        '--noise_ramp_length',
        help='Learning rate rampdown length for projection. ' + \
            'Should be in range [0, 1]. Default: %(default)s',
        default=0.75,
        type=float,
        metavar='VALUE'
    )

    parser.add_argument(
        '--regularize_noise_weight',
        help='The weight for noise regularization. Default: %(default)s',
        default=1e5,
        type=float,
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
        '--num_snapshots',
        help='Number of snapshots. Default: %(default)s',
        type=int,
        default=5,
        metavar='VALUE'
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

#----------------------------------------------------------------------------

def get_arg_parser():
    parser = argparse.ArgumentParser(
        description=_description,
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    range_desc = 'NOTE: This is a single argument, where list ' + \
        'elements are separated by "," and ranges are defined as "a-b". ' + \
        'Only integers are allowed.'

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    project_generated_images_parser = subparsers.add_parser(
        'project_generated_images', help='Project generated images')

    project_generated_images_parser.add_argument(
        '--seeds',
        help='List of random seeds for generating images. ' + \
            'Default: 66,230,389,1518. ' + range_desc,
        type=utils.range_type,
        default=[66, 230, 389, 1518],
        metavar='RANGE'
    )

    project_generated_images_parser.add_argument(
        '--truncation_psi',
        help='Truncation psi. Default: %(default)s',
        type=float,
        default=1.0,
        metavar='VALUE'
    )

    _add_shared_arguments(project_generated_images_parser)

    project_real_images_parser = subparsers.add_parser(
        'project_real_images', help='Project real images')

    project_real_images_parser.add_argument(
        '--data_dir',
        help='Dataset root directory',
        type=str,
        required=True,
        metavar='DIR'
    )

    project_real_images_parser.add_argument(
        '--seed',
        help='When there are more images available than ' + \
            'the number that is going to be projected this ' + \
            'seed is used for picking samples. Default: %(default)s',
        type=int,
        default=1234,
        metavar='VALUE'
    )

    project_real_images_parser.add_argument(
        '--num_images',
        type=int,
        help='Number of images to project. Default: %(default)s',
        default=3,
        metavar='VALUE'
    )

    _add_shared_arguments(project_real_images_parser)

    return parser

#----------------------------------------------------------------------------

def project_images(G, images, name_prefix, args):

    device = torch.device(args.gpu[0] if args.gpu else 'cpu')
    if device.index is not None:
        torch.cuda.set_device(device.index)
    if len(args.gpu) > 1:
        warnings.warn(
            'Multi GPU is not available for projection. ' + \
            'Using device {}'.format(device)
        )
    G = utils.unwrap_module(G).to(device)

    lpips_model = stylegan2.external_models.lpips.LPIPS_VGG16(
        pixel_min=args.pixel_min, pixel_max=args.pixel_max)

    proj = stylegan2.project.Projector(
        G=G,
        dlatent_avg_samples=10000,
        dlatent_avg_label=args.label,
        dlatent_device=device,
        dlatent_batch_size=1024,
        lpips_model=lpips_model,
        lpips_size=256
    )

    for i in range(0, len(images), args.batch_size):
        target = images[i: i + args.batch_size]
        proj.start(
            target=target,
            num_steps=args.num_steps,
            initial_learning_rate=args.initial_learning_rate,
            initial_noise_factor=args.initial_noise_factor,
            lr_rampdown_length=args.lr_rampdown_length,
            lr_rampup_length=args.lr_rampup_length,
            noise_ramp_length=args.noise_ramp_length,
            regularize_noise_weight=args.regularize_noise_weight,
            verbose=True,
            verbose_prefix='Projecting image(s) {}/{}'.format(
                i * args.batch_size + len(target), len(images))
        )
        snapshot_steps = set(
            args.num_steps - np.linspace(
                0, args.num_steps, args.num_snapshots, endpoint=False, dtype=int))
        for k, image in enumerate(
        utils.tensor_to_PIL(target, pixel_min=args.pixel_min, pixel_max=args.pixel_max)):
            image.save(os.path.join(args.output, name_prefix[i + k] + 'target.png'))
        for j in range(args.num_steps):
            proj.step()
            if j in snapshot_steps:
                generated = utils.tensor_to_PIL(
                    proj.generate(), pixel_min=args.pixel_min, pixel_max=args.pixel_max)
                for k, image in enumerate(generated):
                    image.save(os.path.join(
                        args.output, name_prefix[i + k] + 'step%04d.png' % (j + 1)))

#----------------------------------------------------------------------------

def project_generated_images(G, args):
    latent_size, label_size = G.latent_size, G.label_size
    device = torch.device(args.gpu[0] if args.gpu else 'cpu')
    if device.index is not None:
        torch.cuda.set_device(device.index)
    G.to(device)
    if len(args.gpu) > 1:
        warnings.warn(
            'Noise can not be randomized based on the seed ' + \
            'when using more than 1 GPU device. Noise will ' + \
            'now be randomized from default random state.'
        )
        G.random_noise()
        G = torch.nn.DataParallel(G, device_ids=args.gpu)
    else:
        noise_reference = G.static_noise()

    def get_batch(seeds):
        latents = []
        labels = []
        if len(args.gpu) <= 1:
            noise_tensors = [[] for _ in noise_reference]
        for seed in seeds:
            rnd = np.random.RandomState(seed)
            latents.append(torch.from_numpy(rnd.randn(latent_size)))
            if len(args.gpu) <= 1:
                for i, ref in enumerate(noise_reference):
                    noise_tensors[i].append(
                        torch.from_numpy(rnd.randn(*ref.size()[1:])))
            if label_size:
                labels.append(torch.tensor([rnd.randint(0, label_size)]))
        latents = torch.stack(latents, dim=0).to(device=device, dtype=torch.float32)
        if labels:
            labels = torch.cat(labels, dim=0).to(device=device, dtype=torch.int64)
        else:
            labels = None
        if len(args.gpu) <= 1:
            noise_tensors = [
                torch.stack(noise, dim=0).to(device=device, dtype=torch.float32)
                for noise in noise_tensors
            ]
        else:
            noise_tensors = None
        return latents, labels, noise_tensors

    images = []

    progress = utils.ProgressWriter(len(args.seeds))
    progress.write('Generating images...', step=False)

    for i in range(0, len(args.seeds), args.batch_size):
        latents, labels, noise_tensors = get_batch(args.seeds[i: i + args.batch_size])
        if noise_tensors is not None:
            G.static_noise(noise_tensors=noise_tensors)
        with torch.no_grad():
            images.append(G(latents, labels=labels))
        progress.step()

    images = torch.cat(images, dim=0)

    progress.write('Done!', step=False)
    progress.close()

    name_prefix = ['seed%04d-' % seed for seed in args.seeds]
    project_images(G, images, name_prefix, args)

#----------------------------------------------------------------------------

def project_real_images(G, args):
    device = torch.device(args.gpu[0] if args.gpu else 'cpu')
    print('Loading images from "%s"...' % args.data_dir)
    dataset = utils.ImageFolder(
        args.data_dir, pixel_min=args.pixel_min, pixel_max=args.pixel_max)

    rnd = np.random.RandomState(args.seed)
    indices = rnd.choice(
        len(dataset), size=min(args.num_images, len(dataset)), replace=False)
    images = []
    for i in indices:
        data = dataset[i]
        if isinstance(data, (tuple, list)):
            data = data[0]
        images.append(data)
    images = torch.stack(images).to(device)
    name_prefix = ['image%04d-' % i for i in indices]
    print('Done!')
    project_images(G, images, name_prefix, args)

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

    if args.command == 'project_generated_images':
        project_generated_images(G, args)
    elif args.command == 'project_real_images':
        project_real_images(G, args)
    else:
        raise TypeError('Unkown command {}'.format(args.command))


if __name__ == '__main__':
    main()
