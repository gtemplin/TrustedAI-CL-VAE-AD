#!/usr/bin/env python3

import os
import yaml
from src.fuzzy_vae import FuzzyVAE

from copy import deepcopy


def load_config(config_filename: str):

    assert(os.path.exists(config_filename))
    assert(os.path.isfile(config_filename))

    # Load config file
    config = None
    try:
        with open(config_filename, 'r') as ifile:
            config = yaml.safe_load(ifile)

    except IOError as e:
        raise e
    except yaml.YAMLError as e:
        raise e

    return config

def save_config(config: dict, config_filename: str):

    print(config)

    try:
        with open(config_filename, 'w') as ofile:
            yaml.safe_dump(dict(config), ofile)
    except IOError as e:
        raise e
    except yaml.YAMLError as e:
        raise e
    


def load_model(log_dir: str):

    assert(os.path.exists(log_dir))
    assert(os.path.isdir(log_dir))

    config_path = os.path.join(log_dir, 'config.yml')
    config = load_config(config_path)

    # NOTE: TensorFlow modifies 'config' for some reason, so here we pass a deepcopy to force non-reference
    model = FuzzyVAE(deepcopy(config))
    model.load_model(log_dir)

    return model, config