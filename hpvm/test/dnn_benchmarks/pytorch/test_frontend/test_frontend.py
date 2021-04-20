#!/usr/bin/env python3
import shutil
import site
from pathlib import Path
from subprocess import run
from sys import argv

self_folder = Path(__file__).parent.absolute()
site.addsitedir(self_folder.parent)
import dnn

netname = argv[1]
codegen_dir = Path(f"./{netname}")
print(f"Generating {netname} to {codegen_dir}")
if codegen_dir.exists():
    shutil.rmtree(codegen_dir)
target_binary, _ = dnn.export_example_dnn(netname, codegen_dir, False)
run([str(target_binary), "test"], check=True)
