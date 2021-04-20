#!/usr/bin/env python3
import shutil
import site
from pathlib import Path
from sys import argv

from hpvm_profiler import profile_configs, read_hpvm_configs, write_hpvm_configs, plot_hpvm_configs

self_folder = Path(__file__).parent.absolute()
site.addsitedir(self_folder.parent)
import dnn

netname = argv[1]
codegen_dir = Path(f"./{netname}")
if codegen_dir.exists():
    shutil.rmtree(codegen_dir)
binary_file, _ = dnn.export_example_dnn(netname, codegen_dir, False)
config_file = self_folder / "../../hpvm-c/benchmarks" / netname / "data/tuner_confs.txt"
out_config_file = f"./{netname}.txt"
header, configs = read_hpvm_configs(config_file)
profile_configs(binary_file, configs[1:6], configs[0], progress_bar=False)
write_hpvm_configs(header, configs[:6], out_config_file)
plot_hpvm_configs(out_config_file, f"{netname}.png")
