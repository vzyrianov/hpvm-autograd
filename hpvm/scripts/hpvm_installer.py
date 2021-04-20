#!/usr/bin/env python3
from argparse import ArgumentParser
from os import chdir, environ, makedirs
from pathlib import Path
from subprocess import CalledProcessError, check_call
from typing import List, Union

VERSION = "9.0.0"
URL = "http://releases.llvm.org"
DOWNLOADER = "curl"
CLANG_DIR = f"cfe-{VERSION}.src"
CLANG_TARBALL = f"{CLANG_DIR}.tar.xz"
LLVM_DIR = f"llvm-{VERSION}.src"
LLVM_TARBALL = f"{LLVM_DIR}.tar.xz"

ROOT_DIR = (Path(__file__).parent / "..").absolute().resolve()
MODEL_PARAMS_TAR = Path("model_params.tar.gz")
MODEL_PARAMS_DIR = ROOT_DIR / "test/dnn_benchmarks/model_params"
MODEL_PARAMS_LINK = "https://databank.illinois.edu/datafiles/o3izd/download"

LINKS = [
    "CMakeLists.txt",
    "cmake",
    "include",
    "lib",
    "projects",
    "test",
    "tools",
]
MAKE_TARGETS = ["hpvm-clang"]
MAKE_TEST_TARGETS = ["check-hpvm-dnn", "check-hpvm-pass"]

# Relative to project root which is __file__.parent.parent
PY_PACKAGES = [
    "projects/hpvm-profiler",
    "projects/predtuner",
    "projects/torch2hpvm",
    "projects/keras",
]

PYTHON_REQ = ((3, 6), (3, 7))  # This means >= 3.6, < 3.7


def parse_args(args=None):
    parser = ArgumentParser(
        "hpvm_installer", description="Script for automatic HPVM installation."
    )
    parser.add_argument(
        "-m",
        "--no-build",
        action="store_true",
        help="Configure but don't build HPVM. "
        "This will require you to install HPVM manually using cmake and make. "
        "For more details, refer to README.md. Default: False.",
    )
    parser.add_argument(
        "-j",
        "--parallel",
        type=int,
        default=2,
        help="How many threads to build with. This argument is relayed on to 'make'. Default: 2",
    )
    parser.add_argument(
        "-b",
        "--build-dir",
        type=Path,
        default="build",
        help=f"Where to create the build directory "
        "(absolute path, or relative to current directory). Default: build",
    )
    parser.add_argument(
        "-t",
        "--targets",
        type=str,
        default="all",
        help="Build target(s) for LLVM such as X86, ARM. "
        'Use semicolon to separate multiple targets such as "X86;ARM". '
        'Defaults to "all" which is to build all supported targets. '
        "Supported targets: AArch64, AMDGPU, ARM, BPF, Hexagon, Mips, MSP430, NVPTX, PowerPC, "
        "Sparc, SystemZ, X86, XCore.",
    )
    parser.add_argument(
        "--ninja",
        action="store_true",
        help="Use Ninja to build HPVM. Uses 'make' otherwise.",
    )
    parser.add_argument(
        "-r", "--run-tests", action="store_true", help="Build and run test cases"
    )
    parser.add_argument(
        "--no-pypkg", action="store_true", help="Don't build the HPVM Python Packages"
    )
    parser.add_argument(
        "--no-params", action="store_true", help="Don't download DNN model parameters"
    )
    parser.add_argument(
        "cmake_args",
        type=str,
        nargs="*",
        default="",
        help="Argument to relay on to CMake. Separate with space and do not include the dashes. "
        "Example: DCMAKE_BUILD_TYPE=Release DCMAKE_INSTALL_PREFIX=install",
    )
    args = parser.parse_args(args)
    args.cmake_args = [f"-{arg}" for arg in args.cmake_args]
    return args


def prompt_args():
    def parse_yn(s: str):
        table = {"y": True, "n": False}
        return table.get(s)

    def parse_int(s: str):
        try:
            v = int(s)
        except ValueError:
            return None
        if v <= 0:
            return None
        return v

    def parse_targets(s: str):
        if " " in s:
            return None
        return s

    # Use this to get all default arguments
    args = parse_args([])

    print("No Flags found. Using command line prompts.")
    print("Alternatively, please call this script with -h for all available options.")
    print("CLI arguments cover more options than this interactive prompt.")
    auto_build = input_with_check(
        "Build and install HPVM automatically? [y/n]: ", parse_yn, "Please enter y or n"
    )
    args.no_build = not auto_build
    if not auto_build:
        # We no longer need the following fields.
        return args
    args.parallel = input_with_check(
        "Number of threads: ", parse_int, "Please enter a positive integer"
    )
    print(
        "These build targets are supported: AArch64, AMDGPU, ARM, BPF, Hexagon, "
        "Mips, MSP430, NVPTX, PowerPC, Sparc, SystemZ, X86, XCore.\n"
        "If building for multiple targets, seperate options with semicolon:\n"
        "e.g. X86;ARM"
    )
    args.targets = input_with_check(
        "Build target: ", parse_targets, "Input shouldn't contain space"
    )

    print(
        """Additional arguments to CMake? Split by space and no dashes.
Example: "DCMAKE_BUILD_TYPE=Release DCMAKE_INSTALL_PREFIX=install".
Arguments: """
    )
    args.cmake_args = input()
    if args.cmake_args.strip() != "":    
      args.cmake_args = [f"-{arg}" for arg in args.cmake_args.split(" ")]

    args.no_pypkg = not input_with_check(
        "Install HPVM Python Packages (recommended)? [y/n]: ", parse_yn, "Please enter y or n"
    )
    args.no_params = not input_with_check(
        "Download DNN weights (recommended)? [y/n]: ", parse_yn, "Please enter y or n"
    )
    args.run_tests = input_with_check(
        "Build and run tests? [y/n]: ", parse_yn, "Please enter y or n"
    )
    return args


def print_args(args):
    print("Running with the following options:")
    print(f"  Automated build: {not args.no_build}")
    print(f"  Build directory: {args.build_dir}")
    build_sys = "ninja" if args.ninja else "make"
    print(f"  Build system: {build_sys}")
    print(f"  Threads: {args.parallel}")
    print(f"  Targets: {args.targets}")
    print(f"  Download DNN weights: {not args.no_params}")
    print(f"  Run tests: {args.run_tests}")
    print(f"  CMake arguments: {args.cmake_args}")


def check_python_version():
    from sys import version_info, version, executable
    
    lowest, highest = PYTHON_REQ
    if not (lowest <= version_info < highest):
        lowest_str = ".".join([str(n) for n in lowest])
        highest_str = ".".join([str(n) for n in highest])
        version_short_str = ".".join([str(n) for n in version_info])
        raise RuntimeError(
            f"You are using Python {version_short_str}, unsupported by HPVM. "
            f"HPVM requires Python version '{lowest_str} <= version < {highest_str}'.\n"
            f"(Current Python binary: {executable})\n"
            f"Detailed version info:\n{version}"
        )


def check_download_llvm_clang():
    llvm = ROOT_DIR / "llvm"
    if llvm.is_dir():
        print("Found LLVM directory, not extracting it again.")
    else:
        if Path(LLVM_TARBALL).is_file():
            print(f"Found {LLVM_TARBALL}, not downloading it again.")
        else:
            print(f"Downloading {LLVM_TARBALL}...")
            print(f"=============================")
            download(f"{URL}/{VERSION}/{LLVM_TARBALL}", LLVM_TARBALL)
        check_call(["tar", "xf", LLVM_TARBALL])
        check_call(["mv", LLVM_DIR, str(llvm)])
    tools = llvm / "tools"
    assert tools.is_dir(), "Problem with LLVM download. Exiting!"
    if Path(LLVM_TARBALL).is_file():
        Path(LLVM_TARBALL).unlink()  # Remove tarball
    # TODO: check and remove this
    environ["LLVM_SRC_ROOT"] = str(ROOT_DIR / "llvm")

    clang = tools / "clang"
    if clang.is_dir():
        print("Found clang directory, not extracting it again.")
        return
    print(f"Downloading {CLANG_TARBALL}...")
    print(f"=============================")
    download(f"{URL}/{VERSION}/{CLANG_TARBALL}", CLANG_TARBALL)
    check_call(["tar", "xf", CLANG_TARBALL])
    check_call(["mv", CLANG_DIR, str(clang)])
    assert clang.is_dir(), "Problem with clang download. Exiting!"
    if Path(CLANG_TARBALL).is_file():
        Path(CLANG_TARBALL).unlink()


def check_download_model_params():
    if MODEL_PARAMS_DIR.is_dir():
        print("Found model parameters, not extracting it again.")
        return
    if MODEL_PARAMS_TAR.is_file():
        print(f"Found {MODEL_PARAMS_TAR}, not downloading it again.")
    else:
        print(f"Downloading DNN model parameters: {MODEL_PARAMS_TAR}...")
        print(f"=============================")
        download(MODEL_PARAMS_LINK, MODEL_PARAMS_TAR)
    print(
        f"Extracting DNN model parameters {MODEL_PARAMS_TAR} => {MODEL_PARAMS_DIR}..."
    )
    # Decompression is pretty time-consuming so we try to show a progress bar:
    try:
        check_call(f"pv {MODEL_PARAMS_TAR} | tar xz", shell=True)
    except CalledProcessError:
        # Maybe `pv` is not installed. Fine, we'll run without progress bar.
        print(
            ">> 'pv' is not installed, no progress bar will be shown during decompression."
        )
        print(">> Decompression ongoing...")
        check_call(["tar", "xzf", MODEL_PARAMS_TAR])
    check_call(["mv", "model_params", MODEL_PARAMS_DIR])
    if MODEL_PARAMS_TAR.is_file():
        MODEL_PARAMS_TAR.unlink()


def link_and_patch():
    from os import symlink

    cwd = Path.cwd()
    hpvm = ROOT_DIR / "llvm/tools/hpvm"
    print("Adding HPVM sources to tree...")
    makedirs(hpvm, exist_ok=True)
    for link in LINKS:
        if not (hpvm / link).exists():
            print(ROOT_DIR / link, hpvm / link)
            symlink(ROOT_DIR / link, hpvm / link)
    print("Applying HPVM patches...")
    chdir(ROOT_DIR / "llvm_patches")
    check_call(["bash", "./construct_patch.sh"])
    check_call(["bash", "./apply_patch.sh"])
    print("Patches applied.")
    chdir(cwd)


def build(
    build_dir: Path,
    nthreads: int,
    targets: str,
    use_ninja: bool,
    cmake_additional_args: List[str],
):
    print("Now building...")
    print(f"Using {nthreads} threads to build HPVM.")
    makedirs(build_dir, exist_ok=True)

    cwd = Path.cwd()
    chdir(build_dir)
    cmake_args = [
        "cmake",
        str(ROOT_DIR / "llvm"),
        "-DCMAKE_C_COMPILER=gcc",
        "-DCMAKE_CXX_COMPILER=g++",
        f"-DLLVM_TARGETS_TO_BUILD={targets}",
        *cmake_additional_args,
    ]
    if use_ninja:
        cmake_args.append("-GNinja")
    print(f"CMake: {' '.join(cmake_args)}")
    print(f"=============================")
    check_call(cmake_args)

    build_sys = "ninja" if use_ninja else "make"
    build_args = [build_sys, f"-j{nthreads}", *MAKE_TARGETS]
    print(f"Build system ({build_sys}): {' '.join(build_args)}")
    print(f"=============================")
    check_call(build_args)
    chdir(cwd)


def install_py_packages():
    import sys

    for package in PY_PACKAGES:
        package_home = ROOT_DIR / package
        print(f"Installing python package {package_home}")
        check_call([sys.executable, "-m", "pip", "install", str(package_home)])


def run_tests(build_dir: Path, use_ninja: bool, nthreads: int):
    cwd = Path.cwd()
    chdir(build_dir)
    build_sys = "ninja" if use_ninja else "make"
    build_args = [build_sys, f"-j{nthreads}", *MAKE_TARGETS]
    print(f"Tests: {' '.join(build_args)}")
    print(f"=============================")
    check_call(build_args)
    chdir(cwd)


def input_with_check(prompt: str, parse, prompt_when_invalid: str):
    input_str = input(prompt)
    value = parse(input_str)
    while value is None:
        print(f"{prompt_when_invalid}; got {input_str}")
        input_str = input(prompt)
        value = parse(input_str)
    return value


def download(link: str, output: Union[Path, str]):
    check_call(["curl", "-L", link, "-o", str(output)])


def main():
    from sys import argv

    # Don't parse args if no args given -- use prompt mode
    args = prompt_args() if len(argv) == 1 else parse_args()
    if not args.no_pypkg:
        check_python_version()
    print_args(args)
    check_download_llvm_clang()
    link_and_patch()
    if not args.no_params:
        check_download_model_params()
    if not args.no_pypkg:
        install_py_packages()
    if args.no_build:
        print(
            """
HPVM not installed.
To complete installation, follow these instructions:
- Create and navigate to a folder "./build" 
- Run "cmake ../llvm [options]". Find potential options in README.md.
- Run "make -j<number of threads> hpvm-clang" and then "make install"
For more details refer to README.md.
"""
        )
        return
    else:
        build(args.build_dir, args.parallel, args.targets, args.ninja, args.cmake_args)
    if args.run_tests:
        run_tests(args.build_dir, args.ninja, args.parallel)
    else:
        print("Skipping tests.")


if __name__ == "__main__":
    main()
