import warnings
import os
import torch
from torch import multiprocessing as mp
import stylegan2
from stylegan2 import utils
from stylegan2.external_models import inception, lpips
from stylegan2.metrics import fid, ppl

#----------------------------------------------------------------------------

def get_arg_parser():
    parser = utils.ConfigArgumentParser()

    parser.add_argument(
        '--output',
        help='Output directory for model weights.',
        type=str,
        default=None,
        metavar='DIR'
    )

    #----------------------------------------------------------------------------
    # Model options

    parser.add_argument(
        '--latent',
        help='Size of the prior (noise vector). Default: 512',
        type=int,
        default=512,
        metavar='VALUE'
    )

    parser.add_argument(
        '--label',
        help='Number of unique labels. Default: 0',
        type=int,
        default=0,
        metavar='VALUE'
    )

    parser.add_argument(
        '--channels',
        help='Specify the channels for each layer. ' + \
            'Default: [32, 32, 64, 128, 256, 512, 512, 512, 512]',
        nargs='*',
        type=int,
        default=[32, 32, 64, 128, 256, 512, 512, 512, 512],
        metavar='VALUE'
    )

    parser.add_argument(
        '--base_shape',
        help='Data shape of first layer in generator or ' + \
            'last layer in discriminator. Default: (4, 4)',
        nargs=2,
        type=int,
        default=(4, 4),
        metavar='SIZE'
    )

    parser.add_argument(
        '--kernel_size',
        help='Size of conv kernel. Default: 3',
        type=int,
        default=3,
        metavar='SIZE'
    )

    parser.add_argument(
        '--conv_block_size',
        help='Number of layers in a conv block. Default: 2',
        type=int,
        default=2,
        metavar='VALUE'
    )

    parser.add_argument(
        '--pad_once',
        help='Pad filtered convs only once before filter instead of twice. Default: True',
        type=utils.bool_type,
        const=True,
        nargs='?',
        default=True,
        metavar='BOOL'
    )

    parser.add_argument(
        '--pad_mode',
        help='Padding mode for conv layers. Default: constant',
        type=str,
        default='constant',
        metavar='MODE'
    )

    parser.add_argument(
        '--pad_constant',
        help='Padding constant for conv layers when `pad_mode` is \'constant\'. Default: 0',
        type=float,
        default=0,
        metavar='VALUE'
    )

    parser.add_argument(
        '--filter_pad_mode',
        help='Padding mode for filter layers. Default: constant',
        type=str,
        default='constant',
    )

    parser.add_argument(
        '--filter_pad_constant',
        help='Padding constant for filter layers when `filter_pad_mode` is \'constant\'. Default: 0',
        type=float,
        default=0,
    )

    parser.add_argument(
        '--filter',
        help='Filter to use whenever FIR is applied. Default: [1, 3, 3, 1]',
        nargs='*',
        type=float,
        default=[1, 3, 3, 1],
    )

    parser.add_argument(
        '--weight_scale',
        help='Use weight scaling for equalized learning rate. Default: True',
        type=utils.bool_type,
        const=True,
        nargs='?',
        default=True
    )

    #----------------------------------------------------------------------------
    # Generator options

    parser.add_argument(
        '--g_skip',
        help='Use skip connections for the generator. Default: True',
        type=utils.bool_type,
        const=True,
        nargs='?',
        default=True
    )

    parser.add_argument(
        '--g_resnet',
        help='Use resnet connections for the generator. Default: False',
        type=utils.bool_type,
        const=True,
        nargs='?',
        default=False
    )

    parser.add_argument(
        '--g_normalize',
        help='Normalize conv features for generator. Default: True',
        type=utils.bool_type,
        const=True,
        nargs='?',
        default=True
    )

    parser.add_argument(
        '--g_fused_conv',
        help='Fuse conv & upsample into a transposed ' + \
            'conv for the generator. Default: True',
        type=utils.bool_type,
        const=True,
        nargs='?',
        default=True
    )

    parser.add_argument(
        '--g_activation',
        help='The non-linear activaiton function for ' + \
            'the generator. Default: leaky:0.2',
        default='leaky:0.2',
        type=str
    )

    parser.add_argument(
        '--g_conv_resample_mode',
        help='Resample mode for upsampling conv ' + \
            'layers for generator. Default: FIR',
        type=str,
        default='FIR',
    )

    parser.add_argument(
        '--g_skip_resample_mode',
        help='Resample mode for skip connection ' + \
            'upsamples for the generator. Default: FIR',
        type=str,
        default='FIR',
    )

    parser.add_argument(
        '--g_lr',
        help='The learning rate for the generator. Default: 2e-3',
        default=2e-3,
        type=float
    )

    parser.add_argument(
        '--g_betas',
        help='Beta values for the generator Adam optimizer. Default: (0, 0.99)',
        type=float,
        nargs=2,
        default=[0, 0.99],
    )

    parser.add_argument(
        '--g_loss',
        help='Loss function for the generator. Default: logistic_ns',
        default='logistic_ns',
        type=str,
    )

    parser.add_argument(
        '--g_reg',
        help='Regularization function for the generator with an optional weight (:?). Default: pathreg:2',
        default='pathreg:2',
        type=str,
    )

    parser.add_argument(
        '--g_reg_interval',
        help='Interval at which to regularize the generator. Default: 4',
        default=4,
        type=int,
    )

    parser.add_argument(
        '--g_iter',
        help='Number of generator iterations per training iteration. Default: 1',
        default=1,
        type=int,
    )

    parser.add_argument(
        '--style_mix',
        help='The probability of passing more than one ' + \
            'latent to the generator. Default: 0.9',
        type=float,
        default=0.9
    )

    parser.add_argument(
        '--latent_mapping_layers',
        help='The number of layers of the latent mapping network. Default: 8',
        type=int,
        default=8,
    )

    parser.add_argument(
        '--latent_mapping_lr_mul',
        help='The learning rate multiplier for the latent ' + \
            'mapping network. Default: 0.01',
        type=float,
        default=0.01,
    )

    parser.add_argument(
        '--normalize_latent',
        help='Normalize latent inputs. Default: True',
        type=utils.bool_type,
        const=True,
        nargs='?',
        default=True
    )

    parser.add_argument(
        '--modulate_rgb',
        help='Modulate RGB layers. Default: True',
        type=utils.bool_type,
        const=True,
        nargs='?',
        default=True
    )

    #----------------------------------------------------------------------------
    # Discriminator options

    parser.add_argument(
        '--d_skip',
        help='Use skip connections for the discriminator. Default: False',
        type=utils.bool_type,
        const=True,
        nargs='?',
        default=False
    )

    parser.add_argument(
        '--d_resnet',
        help='Use resnet connections for the discriminator. Default: True',
        type=utils.bool_type,
        const=True,
        nargs='?',
        default=True
    )

    parser.add_argument(
        '--d_fused_conv',
        help='Fuse conv & downsample into a strided ' + \
            'conv for the discriminator. Default: True',
        type=utils.bool_type,
        const=True,
        nargs='?',
        default=True
    )

    parser.add_argument(
        '--group_size',
        help='Size of the groups in batch std layer. Default: 4',
        type=int,
        default=4,
    )

    parser.add_argument(
        '--d_activation',
        help='The non-linear activaiton function for the discriminator. Default: leaky:0.2',
        default='leaky:0.2',
        type=str
    )

    parser.add_argument(
        '--d_conv_resample_mode',
        help='Resample mode for downsampling conv ' + \
            'layers for discriminator. Default: FIR',
        type=str,
        default='FIR',
    )

    parser.add_argument(
        '--d_skip_resample_mode',
        help='Resample mode for skip connection ' + \
            'downsamples for the discriminator. Default: FIR',
        type=str,
        default='FIR',
    )

    parser.add_argument(
        '--d_loss',
        help='Loss function for the disriminator. Default: logistic',
        default='logistic',
        type=str,
    )

    parser.add_argument(
        '--d_reg',
        help='Regularization function for the discriminator ' + \
            'with an optional weight (:?). Default: r1:10',
        default='r1:10',
        type=str,
    )

    parser.add_argument(
        '--d_reg_interval',
        help='Interval at which to regularize the discriminator. Default: 16',
        default=16,
        type=int,
    )

    parser.add_argument(
        '--d_iter',
        help='Number of discriminator iterations per training iteration. Default: 1',
        default=1,
        type=int,
    )

    parser.add_argument(
        '--d_lr',
        help='The learning rate for the discriminator. Default: 2e-3',
        default=2e-3,
        type=float
    )

    parser.add_argument(
        '--d_betas',
        help='Beta values for the discriminator Adam optimizer. Default: (0, 0.99)',
        type=float,
        nargs=2,
        default=[0, 0.99],
    )

    #----------------------------------------------------------------------------
    # Training options

    parser.add_argument(
        '--iterations',
        help='Number of iterations to train for. Default: 1 000 000',
        type=int,
        default=1000000,
    )

    parser.add_argument(
        '--gpu',
        help='The cuda device(s) to use. Example: ""--gpu 0 1" will train ' + \
            'on GPU 0 and GPU 1. Default: Only use CPU',
        type=int,
        default=[],
        nargs='*',
    )

    parser.add_argument(
        '--distributed',
        help='When more than one gpu device is passed, automatically ' + \
            'start one process for each device and give it the correct ' + \
            'distributed args (rank, world_size etc). Disable this if ' + \
            'you want training to be performed with only one process ' + \
            'using the DataParallel module. Default: True',
        type=utils.bool_type,
        const=True,
        nargs='?',
        default=True
    )

    parser.add_argument(
        '--rank',
        help='Rank for distributed training.',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--world_size',
        help='World size for distributed training.',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--master_addr',
        help='Address for distributed training.',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--master_port',
        help='Port for distributed training.',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--batch_size',
        help='Size of each batch. Default: 32',
        default=32,
        type=int
    )

    parser.add_argument(
        '--device_batch_size',
        help='Maximum number of items to fit on single device at a time. Default: 4',
        default=4,
        type=int
    )

    parser.add_argument(
        '--g_reg_batch_size',
        help='Size of each batch used to regularize the generator. Default: 16',
        default=16,
        type=int
    )

    parser.add_argument(
        '--g_reg_device_batch_size',
        help='Maximum number of items to fit on single device when ' + \
            'regularizing the generator. Default: 2',
        default=2,
        type=int
    )

    parser.add_argument(
        '--half',
        help='Use mixed precision training. Default: False',
        type=utils.bool_type,
        const=True,
        nargs='?',
        default=False
    )

    parser.add_argument(
        '--resume',
        help='Resume from the latest saved checkpoint in the checkpoint_dir. ' + \
            'This loads all previous training settings except for the dataset options, ' + \
            'device args (--gpu ...) and distributed training args (--rank, --world_size e.t.c) ' + \
            'as well as metrics and logging.',
        type=utils.bool_type,
        const=True,
        nargs='?',
        default=False
    )

    #----------------------------------------------------------------------------
    # Extra metric options

    parser.add_argument(
        '--fid_interval',
        help='Interval of FID evaluations. Default: unused',
        default=None,
        type=int
    )

    parser.add_argument(
        '--ppl_interval',
        help='Interval of PPL evaluations. Default: unused',
        default=None,
        type=int
    )

    parser.add_argument(
        '--ppl_ffhq_crop',
        help='Crop images evaluated for PPL with crop values for FFHQ. Default: False',
        type=utils.bool_type,
        const=True,
        nargs='?',
        default=False
    )

    #----------------------------------------------------------------------------
    # Data options

    parser.add_argument(
        '--pixel_min',
        help='Minimum of the value range of pixels in generated images. Default: -1',
        default=-1,
        type=float
    )

    parser.add_argument(
        '--pixel_max',
        help='Maximum of the value range of pixels in generated images. Default: 1',
        default=1,
        type=float
    )

    parser.add_argument(
        '--data_channels',
        help='Number of channels in the data. Default: 3 (RGB)',
        default=3,
        type=int
    )

    parser.add_argument(
        '--data_dir',
        help='The root directory of the dataset. This argument is required!',
        type=str,
        default=None
    )

    parser.add_argument(
        '--mirror_augment',
        help='Use random horizontal flipping for data images. Default: False',
        type=utils.bool_type,
        const=True,
        nargs='?',
        default=False
    )

    parser.add_argument(
        '--data_workers',
        help='Number of worker processes that handles dataloading. Default: 4',
        default=4,
        type=int
    )

    #----------------------------------------------------------------------------
    # Logging options

    parser.add_argument(
        '--checkpoint_dir',
        help='Save checkpoints to this directory.',
        default=None,
        type=str
    )

    parser.add_argument(
        '--checkpoint_interval',
        help='Save checkpoints with this interval. Default: 10000',
        default=10000,
        type=int
    )

    parser.add_argument(
        '--tensorboard_log_dir',
        help='Log to tensorboard directory.',
        default=None,
        type=str
    )

    parser.add_argument(
        '--tensorboard_image_interval',
        help='Log images to tensorboard with this interval. Default: Unused',
        default=None,
        type=int
    )

    parser.add_argument(
        '--tensorboard_image_size',
        help='Size of images logged to tensorboard. Default: 256',
        default=256,
        type=int
    )

    return parser

#----------------------------------------------------------------------------

def get_dataset(args):
    assert args.data_dir, '--data_dir has to be specified.'
    dataset = utils.ImageFolder(
        args.data_dir,
        mirror=args.mirror_augment,
        pixel_min=args.pixel_min,
        pixel_max=args.pixel_max
    )
    assert len(dataset), 'No images found at {}'.format(args.data_dir)
    return dataset

#----------------------------------------------------------------------------

def get_models(args):
    G_M = stylegan2.models.GeneratorMapping(
        latent_size=args.latent,
        label_size=args.label,
        num_layers=args.latent_mapping_layers,
        hidden=args.latent,
        activation=args.g_activation,
        normalize_input=args.normalize_latent,
        lr_mul=args.latent_mapping_lr_mul,
        weight_scale=args.weight_scale
    )

    common_kwargs = dict(
        data_channels=args.data_channels,
        base_shape=args.base_shape,
        channels=args.channels,
        conv_filter=args.filter,
        skip_filter=args.filter,
        kernel_size=args.kernel_size,
        conv_pad_mode=args.pad_mode,
        conv_pad_constant=args.pad_constant,
        filter_pad_mode=args.filter_pad_mode,
        filter_pad_constant=args.filter_pad_constant,
        pad_once=args.pad_once,
        conv_block_size=args.conv_block_size,
        weight_scale=args.weight_scale
    )

    G_S = stylegan2.models.GeneratorSynthesis(
        latent_size=args.latent,
        demodulate=args.g_normalize,
        modulate_data_out=args.modulate_rgb,
        activation=args.g_activation,
        conv_resample_mode=args.g_conv_resample_mode,
        skip_resample_mode=args.g_skip_resample_mode,
        resnet=args.g_resnet,
        skip=args.g_skip,
        fused_resample=args.g_fused_conv,
        **common_kwargs
    )

    G = stylegan2.models.Generator(G_mapping=G_M, G_synthesis=G_S)

    D = stylegan2.models.Discriminator(
        label_size=args.label,
        activation=args.d_activation,
        conv_resample_mode=args.d_conv_resample_mode,
        skip_resample_mode=args.d_skip_resample_mode,
        mbstd_group_size=args.group_size,
        resnet=args.d_resnet,
        skip=args.d_skip,
        fused_resample=args.d_fused_conv,
        **common_kwargs
    )
    return G, D

#----------------------------------------------------------------------------

def get_trainer(args):
    dataset = get_dataset(args)
    if args.resume and stylegan2.train._find_checkpoint(args.checkpoint_dir):
        trainer = stylegan2.train.Trainer.load_checkpoint(
            args.checkpoint_dir,
            dataset,
            device=args.gpu,
            rank=args.rank,
            world_size=args.world_size,
            master_addr=args.master_addr,
            master_port=args.master_port,
            tensorboard_log_dir=args.tensorboard_log_dir
        )
    else:
        G, D = get_models(args)
        trainer = stylegan2.train.Trainer(
            G=G,
            D=D,
            latent_size=args.latent,
            dataset=dataset,
            device=args.gpu,
            batch_size=args.batch_size,
            device_batch_size=args.device_batch_size,
            label_size=args.label,
            data_workers=args.data_workers,
            G_loss=args.g_loss,
            D_loss=args.d_loss,
            G_reg=args.g_reg,
            G_reg_interval=args.g_reg_interval,
            G_opt_kwargs={'lr': args.g_lr, 'betas': args.g_betas},
            G_reg_batch_size=args.g_reg_batch_size,
            G_reg_device_batch_size=args.g_reg_device_batch_size,
            D_reg=args.d_reg,
            D_reg_interval=args.d_reg_interval,
            D_opt_kwargs={'lr': args.d_lr, 'betas': args.d_betas},
            style_mix_prob=args.style_mix,
            G_iter=args.g_iter,
            D_iter=args.d_iter,
            tensorboard_log_dir=args.tensorboard_log_dir,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
            half=args.half,
            rank=args.rank,
            world_size=args.world_size,
            master_addr=args.master_addr,
            master_port=args.master_port
        )
    if args.fid_interval and not args.rank:
        fid_model = inception.InceptionV3FeatureExtractor(
            pixel_min=args.pixel_min, pixel_max=args.pixel_max)
        trainer.register_metric(
            name='FID (299x299)',
            eval_fn=fid.FID(
                trainer.Gs,
                trainer.prior_generator,
                dataset=dataset,
                fid_model=fid_model,
                fid_size=299,
                reals_batch_size=64
            ),
            interval=args.fid_interval
        )
        trainer.register_metric(
            name='FID',
            eval_fn=fid.FID(
                trainer.Gs,
                trainer.prior_generator,
                dataset=dataset,
                fid_model=fid_model,
                fid_size=None
            ),
            interval=args.fid_interval
        )
    if args.ppl_interval and not args.rank:
        lpips_model = lpips.LPIPS_VGG16(
            pixel_min=args.pixel_min, pixel_max=args.pixel_max)
        crop = None
        if args.ppl_ffhq_crop:
            crop = ppl.PPL.FFHQ_CROP
        trainer.register_metric(
            name='PPL_end',
            eval_fn=ppl.PPL(
                trainer.Gs,
                trainer.prior_generator,
                full_sampling=False,
                crop=crop,
                lpips_model=lpips_model,
                lpips_size=256
            ),
            interval=args.ppl_interval
        )
        trainer.register_metric(
            name='PPL_full',
            eval_fn=ppl.PPL(
                trainer.Gs,
                trainer.prior_generator,
                full_sampling=True,
                crop=crop,
                lpips_model=lpips_model,
                lpips_size=256
            ),
            interval=args.ppl_interval
        )
    if args.tensorboard_image_interval:
        for static in [True, False]:
            for trunc in [0.5, 0.7, 1.0]:
                if static:
                    name = 'static'
                else:
                    name = 'random'
                name += '/trunc_{:.1f}'.format(trunc)
                trainer.add_tensorboard_image_logging(
                    name=name,
                    num_images=4,
                    interval=args.tensorboard_image_interval,
                    resize=args.tensorboard_image_size,
                    seed=1234567890 if static else None,
                    truncation_psi=trunc,
                    pixel_min=args.pixel_min,
                    pixel_max=args.pixel_max
                )
    return trainer

#----------------------------------------------------------------------------

def run(args):
    if not args.rank:
        if not (args.checkpoint_dir or args.output):
            warnings.warn(
                'Neither an output path or checkpoint dir has been ' + \
                'given. Weights from this training run will never ' + \
                'be saved.'
            )
        if args.output:
            assert os.path.isdir(args.output) or not os.path.splitext(args.output)[-1], \
                '--output argument should specify a directory, not a file.'
    trainer = get_trainer(args)
    trainer.train(iterations=args.iterations)
    if not args.rank and args.output:
        print('Saving models to {}'.format(args.output))
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        for model_name in ['G', 'D', 'Gs']:
            getattr(trainer, model_name).save(
                os.path.join(args.output_dir, model_name + '.pth'))

#----------------------------------------------------------------------------

def run_distributed(rank, args):
    args.rank = rank
    args.world_size = len(args.gpu)
    args.gpu = args.gpu[rank]
    args.master_addr = args.master_addr or '127.0.0.1'
    args.master_port = args.master_port or '23456'
    run(args)

#----------------------------------------------------------------------------

def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    if len(args.gpu) > 1 and args.distributed:
        assert args.rank is None and args.world_size is None, \
            'When --distributed is enabled (default) the rank and ' + \
            'world size can not be given as this is set up automatically. ' + \
            'Use --distributed 0 to disable automatic setup of distributed training.'
        mp.spawn(run_distributed, nprocs=len(args.gpu), args=(args,))
    else:
        run(args)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    main()
