import argparse
from omegaconf import OmegaConf


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
        "--cfg-path", default="config.yaml", help="path to configuration file."
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


def get_config():
    return Config(parse_args()).config


if __name__ == "__main__":
    config = get_config()
    print(config)
