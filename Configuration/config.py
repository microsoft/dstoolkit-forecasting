import yaml
from box import Box
import sys
import os

from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent

root = get_project_root()
with open(os.path.join(root, "Configuration/config.yaml"), "r") as ymlfile:
  cfg_path = Box(yaml.safe_load(ymlfile))

