"""Portable project settings; no repository-specific paths in generated defaults."""

import copy
import fnmatch
import json
from pathlib import Path

DEFAULT = {
    "schema": 1,
    "toolkit": "cuda12",
    "source_dirs": ["."],
    "exclude": ["build/**", ".git/**", ".xedt/**", ".venv/**"],
    "include_dirs": [".", "include"],
    "standard": "c++20",
    "gpu_arch": "sm_80",
    "cuda_headers": ["*.cuh"],
    "dependencies": [],
    "defines": [],
    "extra_flags": [],
    "file_rules": [],
    "check_files": [],
}


def defaults(preset=None):
    config = copy.deepcopy(DEFAULT)
    if preset == "fused-mcl":
        config.update({
            "toolkit": "cuda12",
            "source_dirs": ["fusedmcl", "benchmarks", "examples"],
            "include_dirs": [".", "fusedmcl"],
            "cuda_headers": ["*.cuh", "fusedmcl/*.hpp"],
            "dependencies": ["mpi", "openmp", "zstd", "fmt", "cxxopts"],
            "file_rules": [
                {"match": "*/retained_multistage.cu", "defines": ["FMCL_TILE_VALUE_FP64=1"]},
                {"match": "*/streaming_multistage.cu", "defines": ["FMCL_TILE_VALUE_FP64=1"]},
                {"match": "*/ocean_multistage.cu", "defines": ["FMCL_TILE_VALUE_FP64=1", "QUIET"]},
                {"match": "benchmarks/single_spgemm/main.cu", "defines": ["FMCL_TILE_VALUE_FP64=1", "QUIET"]},
            ],
            "check_files": [
                "fusedmcl/localspgemm/retained_hash_product.cu",
                "fusedmcl/localspgemm/streaming_hash_spgemm.cu",
                "fusedmcl/dist/engine/mcl_2d_engine.cuh",
                "fusedmcl/main/main_mcl_2d_summa.cu",
                "fusedmcl/kernels/local_spgemm.cuh",
                "fusedmcl/dist/pruning/compact_histogram.cuh",
                "benchmarks/compact_histogram_test.cu",
                "fusedmcl/tools/mtx2bcsc.cpp",
            ],
        })
    return config


def matches(relative, patterns):
    return any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns)


def load(path):
    data = json.loads(path.read_text())
    unknown = set(data) - set(DEFAULT)
    if unknown:
        raise ValueError("Unknown settings: " + ", ".join(sorted(unknown)))
    config = defaults()
    config.update(data)
    if config["schema"] != 1 or config["toolkit"] not in ("cuda12", "cuda13"):
        raise ValueError("Use schema 1 and toolkit cuda12 or cuda13")
    list_keys = ("source_dirs", "exclude", "include_dirs", "cuda_headers", "dependencies",
                 "defines", "extra_flags", "file_rules", "check_files")
    for key in list_keys:
        if not isinstance(config[key], list):
            raise ValueError(f"{key} must be a list")
        if key != "file_rules" and not all(isinstance(x, str) for x in config[key]):
            raise ValueError(f"{key} must contain strings")
    for rule in config["file_rules"]:
        if not isinstance(rule, dict) or not isinstance(rule.get("match"), str):
            raise ValueError("Each file rule needs a match pattern")
        if set(rule) - {"match", "defines", "include_dirs", "extra_flags", "language"}:
            raise ValueError("Unknown file rule setting")
        if rule.get("language", "cuda") not in ("cuda", "c++", "c"):
            raise ValueError("File rule language must be cuda, c++, or c")
        for key in ("defines", "include_dirs", "extra_flags"):
            values = rule.get(key, [])
            if not isinstance(values, list) or not all(isinstance(x, str) for x in values):
                raise ValueError(f"File rule {key} must contain strings")
    if not isinstance(config["standard"], str) or not isinstance(config["gpu_arch"], str):
        raise ValueError("standard and gpu_arch must be strings")
    return config


def sources(root, config):
    paths = set()
    for directory in config["source_dirs"]:
        folder = root / directory
        if not folder.is_dir():
            raise ValueError(f"Source directory does not exist: {folder}")
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix not in (".cu", ".cuh", ".cpp", ".cc", ".cxx", ".hpp", ".h", ".c"):
                continue
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                raise ValueError(f"Source directory must be inside the project: {folder}")
            hidden = any(part.startswith(".") for part in Path(relative).parts)
            if not hidden and not matches(relative, config["exclude"]):
                paths.add(path)
    if not paths:
        raise ValueError("No C++/CUDA files found; configure source_dirs")
    return sorted(paths)
