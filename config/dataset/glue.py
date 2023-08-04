"""
This module provides functionality for generating configuration files for the GLUE benchmark datasets.

This module uses the OmegaConf library to load a base configuration file from `glue.yaml` and generate 
a separate configuration file for each GLUE dataset. The generated configuration files are saved in the 
same directory as `glue.yaml` with the name `glue-{dataset_name}.yaml`.
"""
from omegaconf import OmegaConf
from pathlib import Path

base_cfg = OmegaConf.load(Path(__file__).parent / "glue.yaml")
for name in [
    "cola",
    "sst2",
    "mrpc",
    "qqp",
    "stsb",
    "mnli",
    "mnli_mismatched",
    "mnli_matched",
    "qnli",
    "rte",
    "wnli",
    "ax",
]:
    base_cfg.datasets.name = name
    OmegaConf.save(base_cfg, Path(__file__).parent / f"glue-{name}.yaml")
