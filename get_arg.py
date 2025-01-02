import argparse
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="path to the yaml config file")

parser.add_argument('-mp', '--model_params', default='', nargs='*', help='list of key=value pairs of model options')

args, extras = parser.parse_known_args()

# override default config from cli
opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras),
                      OmegaConf.create({"model_params": args.model_params}))

