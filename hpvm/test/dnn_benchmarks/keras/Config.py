
import pathlib


# Path Relative to Model Params Directory
CUR_SRC_PATH = str(pathlib.Path(__file__).parent.absolute())
MODEL_PARAMS_DIR = CUR_SRC_PATH + "/../../../../hpvm/test/dnn_benchmarks/model_params/"




if __name__ == "__main__":

    abs_path = pathlib.Path(__file__).parent.absolute()
    print (abs_path)
