"""
This script loads the base configuration from the super_glue.yaml file and creates a new configuration file for each dataset specified in the for loop. The new configuration files are saved in the same directory as the base configuration file with the dataset name appended to the filename.

Example:
    If the base configuration file is named super_glue.yaml, running this script will create new configuration files named super_glue-boolq.yaml, super_glue-cb.yaml, super_glue-copa.yaml, and so on for each dataset specified in the for loop.
"""
from pathlib import Path

from omegaconf import OmegaConf

base_cfg = OmegaConf.load(Path(__file__).parent / "super_glue.yaml")
for name in [
    "boolq",
    "cb",
    "copa",
    "multirc",
    "record",
    "rte",
    "wic",
    "wsc",
    "wsc.fixed",
    "axb",
    "axg",
]:
    base_cfg.datasets.name = name
    OmegaConf.save(base_cfg, Path(__file__).parent / f"super_glue-{name}.yaml")
