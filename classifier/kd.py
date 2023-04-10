import argparse, yaml, os

from src import main

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--img', type=str, required=True)
parser.add_argument('--lbl', type=str, required=True)
args = parser.parse_args()

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

with open(args.cfg, 'r') as file:
    cfg = yaml.safe_load(file)

cfg = AttributeDict(cfg)
os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.GPU}"

inferencer = main(cfg, "kd")
inferencer.inference_kd(args.img, args.lbl)