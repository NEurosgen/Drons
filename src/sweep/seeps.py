from itertools import product
from copy import deepcopy
from pathlib import Path

def _assert_override_keys_exist(cfg,param_grid):
    for key in param_grid.keys():
        parts = key.split('.')
        node = cfg
        for p in parts[:-1]:
            assert p in node ,f"Invalid override {key}: sectioon {p} not in config"
            node = node[p]
        last = parts[-1]
        assert last in node , f"Invalid override{key}: {last} not found in config section {'.'.join(parts[:-1])}"