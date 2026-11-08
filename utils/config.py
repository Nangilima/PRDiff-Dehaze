import argparse

import yaml


def dict_to_namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        setattr(namespace, key, dict_to_namespace(value) if isinstance(value, dict) else value)
    return namespace


def load_config(config_arg, default_file):
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--{config_arg}", default=default_file, type=str)
    args = parser.parse_args()

    path = getattr(args, config_arg)
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit(f"config file not found: {path}")
    except yaml.YAMLError as exc:
        raise SystemExit(f"failed to parse config file: {exc}")

    return dict_to_namespace(config)
