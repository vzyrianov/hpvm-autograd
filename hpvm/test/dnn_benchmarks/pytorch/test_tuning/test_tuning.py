#!/usr/bin/env python3
import shutil
import site
from pathlib import Path
from sys import argv

from predtuner import PipedBinaryApp, config_pylogger

self_folder = Path(__file__).parent.absolute()
site.addsitedir(self_folder.parent)
import dnn

# Set up logger
msg_logger = config_pylogger(output_dir=".", verbose=True)


def main():
    netname, is_pred = argv[1:]
    is_pred = int(is_pred)
    # Generating tunable binary
    codegen_dir = Path(f"./{netname}")
    if codegen_dir.exists():
        shutil.rmtree(codegen_dir)
    binary_file, exporter = dnn.export_example_dnn(netname, codegen_dir, True)
    metadata_file = codegen_dir / exporter.metadata_file_name
    # Tuning
    app = PipedBinaryApp("test", binary_file, metadata_file, target_device="cpu")
    tuner = app.get_tuner()
    tuner.tune(
        5,
        3.0,
        is_threshold_relative=True,
        cost_model="cost_linear",
        qos_model="qos_p1" if is_pred else None,
    )
    tuner.dump_configs("configs.json")
    fig = tuner.plot_configs(show_qos_loss=True)
    fig.savefig("configs.png", dpi=300)
    app.dump_hpvm_configs(tuner.best_configs, "hpvm_confs.txt")


if __name__ == "__main__":
    main()
