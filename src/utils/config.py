import argparse
from omegaconf import OmegaConf
import json


class Config:
    def __init__(self, args):
        self.config = OmegaConf.merge(
            OmegaConf.load(args.cfg_path), self._build_user_config(args.options)
        )

    def _build_user_config(self, opts):
        return OmegaConf.from_dotlist([] if opts is None else opts)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-path", default="eval_config.yaml", help="path to configuration file."
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    return args


def flatten_config(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_config(train=True):
    return Config(parse_args()).config


def dump_config(config, path, flatten=False):
    dict_config = OmegaConf.to_container(config)
    if flatten:
        dict_config = flatten_config(dict_config)
    with open(path, "w") as f:
        json.dump(dict_config, f, indent=4)


if __name__ == "__main__":
    config = get_config()
    print(config)
