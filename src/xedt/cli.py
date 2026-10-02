"""The xedt command line."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import __version__
from .config import defaults, load
from .editors import nvim
from .environment import discover, environment_path, packages, prepare
from .project import activate, atomic_write, check, database, preflight, smoke


def parser():
    result = argparse.ArgumentParser(description="xedt — Accelerator Editor: macOS CUDA/C++ editor setup")
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command", required=True)
    for name, help_text in (("init", "create portable project settings"),
                            ("doctor", "inspect compiler, dependencies and database ownership"),
                            ("setup", "prepare headers, probe compiler, and generate database"),
                            ("check", "check real files with both compiler and clangd")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--project", type=Path, default=Path.cwd())
        command.add_argument("--config", type=Path, help="external JSON settings (default: PROJECT/.xedt.json)")
        if name == "init":
            command.add_argument("--preset", choices=["fused-mcl"])
            command.add_argument("--toolkit", choices=["cuda12", "cuda13"],
                                 help="header profile: CUDA 12.4 (default) or CUDA 13.2 with CCCL 3.4.3")
        if name in ("doctor", "setup"):
            command.add_argument("--clangd", help="native clangd path")
            command.add_argument("--clangxx", help="matching native clang++ path")
            command.add_argument("--cache-dir", type=Path, default=Path.home() / "Library/Caches/xedt")
            command.add_argument("--clion", action="store_true", help="also create an owned root database copy")
            command.add_argument("--no-activate", action="store_true", help="preserve existing build database; generate only .xedt/")
        if name == "setup":
            command.add_argument("--reuse-downloads", type=Path, help="import verified archives from an existing cache")
            command.add_argument("--offline", action="store_true", help="require verified cached archives")
            command.add_argument("--check", action="store_true", help="also validate configured check_files")
        if name == "check":
            command.add_argument("files", nargs="*", help="project-relative paths; defaults to check_files")
    editor = commands.add_parser("nvim", help="print or install the Neovim integration module")
    editor.add_argument("--install", action="store_true", help="install module; print the line to load it")
    editor.add_argument("--config-dir", type=Path, default=Path.home() / ".config/nvim")
    return result


def execute(args):
    if args.command == "nvim":
        nvim(args.config_dir.expanduser().resolve(), args.install)
        return
    root = args.project.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Project directory does not exist: {root}")
    settings = args.config.expanduser().resolve() if args.config else root / ".xedt.json"
    output = root / ".xedt"
    if args.command == "init":
        if settings.exists() or settings.is_symlink():
            raise ValueError(f"Settings already exist: {settings}; edit them directly")
        config = defaults(args.preset)
        if args.toolkit:
            config["toolkit"] = args.toolkit
        atomic_write(settings, json.dumps(config, indent=2) + "\n")
        print(f"Created {settings}; review source_dirs, include_dirs, dependencies, and check_files")
        return
    if not settings.is_file():
        raise ValueError(f"Missing settings: {settings}; run xedt init --project {root}")
    config = load(settings)
    if args.command == "check":
        check(root, output, args.files or config["check_files"])
        return
    toolchain = discover(config["toolkit"], args.clangd, args.clangxx)
    selected = packages(config)
    cache = args.cache_dir.expanduser().resolve()
    preflight(root, output, not args.no_activate, args.clion)
    if args.command == "doctor":
        print(f"clangd: {toolchain.clangd} (Clang {toolchain.major})")
        print(f"clang++: {toolchain.clangxx}")
        print(f"Toolkit: {config['toolkit']}; packages: {', '.join(item['name'] for item in selected)}")
        print(f"Environment: {environment_path(cache, selected)}")
        print("Database ownership: OK")
        print("Doctor does not download headers or prove compiler compatibility; setup runs a syntax probe.")
        return
    if args.check and not config["check_files"]:
        raise ValueError("Configure check_files before using setup --check")
    # Detect missing source folders before downloading any package.
    from .config import sources
    sources(root, config)
    environment = prepare(cache, selected,
                          args.reuse_downloads.expanduser().resolve() if args.reuse_downloads else None,
                          args.offline)
    smoke(root, output, config, toolchain, environment)
    entries = database(root, config, toolchain, environment)
    activate(root, output, entries, toolchain, environment, not args.no_activate, args.clion)
    if args.check:
        check(root, output, config["check_files"])


def main(argv=None):
    try:
        execute(parser().parse_args(argv))
    except (ValueError, OSError, subprocess.CalledProcessError, StopIteration) as error:
        print(f"xedt: {error}", file=sys.stderr)
        raise SystemExit(1)
