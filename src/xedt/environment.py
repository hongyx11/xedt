"""Discover native Clang and unpack checksum-pinned Linux headers."""

import hashlib
import json
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"
CORE = ["libgcc-12-dev", "libstdc++-12-dev", "libc6-dev", "linux-libc-dev",
        "cuda_cudart", "cuda_nvcc", "cuda_cccl", "libcurand"]
DEPENDENCIES = {
    "mpi": "libopenmpi-dev", "openmp": "libomp-14-dev", "zstd": "libzstd-dev",
    "fmt": "fmt", "cxxopts": "cxxopts", "cusparse": "libcusparse",
    "eigen": "eigen", "boost_preprocessor": "boost_preprocessor",
    "fast_matrix_market": "fast_matrix_market",
}


def run(*args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, **kwargs)


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class Toolchain:
    clangd: str
    clangxx: str
    resource: Path
    major: int


def version(executable):
    output = subprocess.check_output([str(executable), "--version"], text=True)
    match = re.search(r"(?:clangd|clang) version (\d+)", output)
    if not match:
        raise ValueError(f"Cannot identify Clang version: {executable}")
    return int(match.group(1))


def discover(toolkit, explicit=None, compiler=None):
    if platform.system() != "Darwin":
        raise ValueError("Header setup requires macOS; Linux projects should use their build database")
    minimum = 22 if toolkit == "cuda13" else 17
    candidates = []
    if explicit:
        candidates.append(shutil.which(explicit) or explicit)
    else:
        # Neovim may belong to a different Homebrew installation than PATH's brew.
        nvim = shutil.which("nvim")
        brews = [str(Path(nvim).parent / "brew")] if nvim else []
        brews += [shutil.which("brew") or "brew"]
        for brew in dict.fromkeys(brews):
            if not shutil.which(brew):
                continue
            result = subprocess.run([brew, "--prefix", "llvm"], text=True, capture_output=True)
            if result.returncode == 0:
                candidates.append(str(Path(result.stdout.strip()) / "bin/clangd"))
        candidates += ["/opt/homebrew/opt/llvm/bin/clangd", "/usr/local/opt/llvm/bin/clangd",
                       shutil.which("clangd") or ""]
    errors = []
    for candidate in dict.fromkeys(candidates):
        if not Path(candidate).is_file():
            errors.append(f"not found: {candidate}")
            continue
        try:
            major = version(candidate)
            clangxx = (shutil.which(compiler) or compiler) if compiler else str(Path(candidate).with_name("clang++"))
            if major < minimum:
                raise ValueError(f"{toolkit} needs Clang {minimum}+; found {major}")
            if version(clangxx) != major:
                raise ValueError("clangd and clang++ must have matching major versions")
            resource = Path(subprocess.check_output([clangxx, "-print-resource-dir"], text=True).strip())
            if not (resource / "include/cuda_wrappers").is_dir():
                raise ValueError(f"Missing Clang CUDA wrappers: {resource}")
            return Toolchain(candidate, clangxx, resource, major)
        except (ValueError, OSError, subprocess.CalledProcessError) as error:
            errors.append(f"{candidate}: {error}")
    raise ValueError("No compatible native Clang installation. Use --clangd PATH.\n" + "\n".join(errors))


def packages(config):
    lock = json.loads((ASSETS / (config["toolkit"] + ".json")).read_text())
    names = list(CORE)
    if config["toolkit"] == "cuda13":
        names.append("cuda_crt")
    for dependency in config["dependencies"]:
        if dependency not in DEPENDENCIES:
            raise ValueError(f"Unknown dependency: {dependency}")
        names.append(DEPENDENCIES[dependency])
    indexed = {item["name"]: item for item in lock}
    missing = set(names) - set(indexed)
    if missing:
        raise ValueError(f"Dependencies unavailable in {config['toolkit']} lock: {', '.join(sorted(missing))}")
    return [indexed[name] for name in dict.fromkeys(names)]


def environment_path(cache, selected):
    identity = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()[:16]
    return cache / "environments" / identity


def archive_for(item, cache, reuse=None, offline=False):
    directory = cache / "downloads"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / item["sha256"]
    if target.is_file() and digest(target) == item["sha256"]:
        return target
    if reuse:
        # Support both the original script's cache and upstream archive filenames.
        for name in (item["name"], item["url"].rsplit("/", 1)[-1], item["sha256"]):
            candidate = reuse / name
            if candidate.is_file() and digest(candidate) == item["sha256"]:
                shutil.copyfile(candidate, target)
                return target
    if offline:
        raise ValueError(f"No verified cached archive for {item['name']} (offline mode)")
    with tempfile.TemporaryDirectory(dir=directory) as staging:
        partial = Path(staging) / "download"
        run("curl", "--fail", "--location", "--silent", "--show-error", "--retry", "3",
            item["url"], "-o", partial)
        if digest(partial) != item["sha256"]:
            raise ValueError(f"Checksum mismatch: {item['name']}")
        partial.replace(target)
    return target


def prepare(cache, selected, reuse=None, offline=False):
    destination = environment_path(cache, selected)
    if (destination / "ready.json").is_file():
        # Verify archives on reruns too; never trust names or downloaded timestamps.
        for item in selected:
            archive_for(item, cache, reuse, offline)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary) / "environment"
        stage.mkdir()
        for item in selected:
            archive = archive_for(item, cache, reuse, offline)
            kind = item["kind"]
            output = stage / ("sysroot" if kind == "deb" else "cuda" if kind == "cuda" else item["name"])
            output.mkdir(parents=True, exist_ok=True)
            if kind == "deb":
                members = subprocess.check_output(["ar", "t", str(archive)], text=True).splitlines()
                member = next(m for m in members if m.startswith("data.tar"))
                data = subprocess.check_output(["ar", "p", str(archive), member])
                run("tar", "-xf", "-", "-C", output, input=data)
            elif kind in ("cuda", "source"):
                run("tar", "-xf", archive, "--strip-components=1", "-C", output)
            else:
                raise ValueError(f"Unknown archive kind: {kind}")
            print(f"Prepared {item['name']}", flush=True)
        (stage / "ready.json").write_text(json.dumps(selected, indent=2) + "\n")
        # An incomplete previous extraction must never be considered a usable SDK.
        if destination.exists():
            raise ValueError(f"Incomplete environment exists: {destination}; move it aside and rerun")
        stage.rename(destination)
    return destination
