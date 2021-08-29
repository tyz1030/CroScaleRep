# Description:
# author TIANYI ZHANG
# version 0.1 Jan 2021
# Copyright (c) 2021 TIANYI ZHANG, DROP Lab, University of Michigan, Ann Arbor

#!/usr/bin/env python3

from sampler.sampler_micro import MicroSamplerGMapChip, MicroSamplerTiffChip
from sampler.sampler_tele import TeleSamplerTiffDense
from sampler import MicroSamplerGMap, MicroSamplerTiff
from sampler import TeleSamplerTiff
from sampler import CSHSampler


def micro_sampler_factory(args):
    if args.microscope_src == "google_map_static_api":
        return MicroSamplerGMap(api_key=args.gmap_api_key, secret=args.url_secret, mode = args.sample_mode, step = args.pix_select_step)
    elif args.microscope_src == "google_map_static_api_and_tile":
        return MicroSamplerGMapChip(api_key=args.gmap_api_key, secret=args.url_secret, mode = args.sample_mode, step = args.pix_select_step)
    elif args.microscope_src == "high_res_telescope":
        return MicroSamplerTiff(white_check = args.white_check_micro, mode = args.sample_mode, step = args.pix_select_step)
    elif args.microscope_src == "high_res_telescope_and_tile":
        return MicroSamplerTiffChip(white_check = args.white_check_micro, mode = args.sample_mode, step = args.pix_select_step)


def tele_sampler_factory(args):
    if args.telescope_src == "tiff":
        return TeleSamplerTiff(args)
    elif args.telescope_src == "tifftile":
        return TeleSamplerTiffDense(args)


def config_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description='Config for sampling from different data source')
    parser.add_argument(
        '--data', type=str, help='dataset. path to telescope data or sample config. Sampling starts with telescope.')
    parser.add_argument(
        '--data_type', type=str, help='.tif, .tiff or .vrt')
    parser.add_argument(
        '--scale_down', type=int, default=1, help='if we are sampling from high res tele scope image, we want to sample it down.')
    parser.add_argument(
        '--flip_axis', type=int, help='some tiff file need axis flip. 1 horzion, 0 vertical, -1 both. see opencv flipcode')
    parser.add_argument(
        '--output', type=str, help='path to which output structured cross-scope samples. Must be empty dir to prevent overwrite')
    parser.add_argument('--microscope_src', type=str,
                        help='google_map_static_api or high_res_telescope')
    parser.add_argument('--telescope_src', type=str,
                        help='tiff or google_earth_engine')
    
    parser.add_argument('--gmap_api_key', type=str,
                        help='key for google map static api')
    parser.add_argument('--url_secret', type=str,
                        help='use with google map api. URL signing secret')
    
    parser.add_argument('--num_teles', type=int,
                        help='number of telescope sample')
    parser.add_argument('--num_micros', type=int,
                        help='number of microscope sample per telescope sample')
    
    parser.add_argument('--white_check_tele', type=int,
                        help='some tif file has white area which is out of the boundry of the map. this option will check if a sample have white area more than xxx %')
    parser.add_argument('--white_check_micro', type=int,
                        help='some tif file has white area which is out of the boundry of the map. this option will check if a sample have white area more than xxx %')
      
    parser.add_argument('--sample_mode', type=str,
                        help='random, line or dense. 1 is random sample in an area, 2 is sample on a straight line, which simulates a trajectory, 3 is samples dense cover the area')
    parser.add_argument('--pix_select_step', type=int, default = 8,
                        help='')
    parser.add_argument('--flip_channels', type=int, default = 0, help='')
    return parser


def main():
    parser = config_parser()
    args = parser.parse_args()

    # Create Microscope Sampler
    micro_sampler = micro_sampler_factory(args)

    # Create Telescope Sampler (A Microscper Sampler will be passed to and owned by Telescope Sampler)
    tele_sampler = tele_sampler_factory(args)

    # cross-scope heirarchical sampler
    croscope_sampler = CSHSampler(
        tele_sampler, micro_sampler, args.output, num_teles=args.num_teles, num_micros=args.num_micros)

    # Run Sampling
    croscope_sampler.sample()
    return


if __name__ == '__main__':
    main()