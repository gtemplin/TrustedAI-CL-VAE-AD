#!/usr/bin/env python3

import os
import yaml
from src.fuzzy_vae import FuzzyVAE

from copy import deepcopy


def import_vae_based_on_type(vae_type: str):

    AVAILABLE_TYPES = [
        'KLGaussian',
        'KurtosisGlobal',
        'KurtosisSingle',
    ]

    if vae_type is not None:
        if vae_type not in AVAILABLE_TYPES:
            raise Exception(f'Error, type {vae_type} not found in available types: {AVAILABLE_TYPES}')
        
        if vae_type.lower() == 'klgaussian':
            raise NotImplementedError('KLGaussian not yet implemented')
        elif vae_type.lower() == 'kurtosisglobal':
            from src.fuzzy_vae import FuzzyVAE
            return FuzzyVAE
        elif vae_type.lower() == 'kurtosissingle':
            raise NotImplementedError('KurtosisSingle not yet implemented')
    else:
        from src.fuzzy_vae import FuzzyVAE
        return FuzzyVAE


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
    model = import_vae_based_on_type(config['model'].get('type'))(deepcopy(config))
    model.load_model(log_dir)

    return model, config