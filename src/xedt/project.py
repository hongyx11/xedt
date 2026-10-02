"""Generate commands, validate real files, and manage only owned artifacts."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from .config import matches, sources
from .environment import digest


def flags(root, config, toolchain, environment, language, relative=""):
    sysroot = environment / "sysroot"
    args = [toolchain.clangxx, f"-resource-dir={toolchain.resource}",
            "--target=x86_64-linux-gnu", f"--sysroot={sysroot}"]
    if language != "c":
        args += ["-nostdinc++", "-std=" + config["standard"]]
    if language == "cuda":
        args += ["-isystem", str(toolchain.resource / "include/cuda_wrappers"),
                 "--cuda-path=" + str(environment / "cuda"), "--cuda-host-only", "-nocudalib",
                 "--cuda-gpu-arch=" + config["gpu_arch"], "-Wno-invalid-constexpr"]
    else:
        args += ["-isystem", str(environment / "cuda/include")]
    cccl = environment / "cuda/include/cccl"
    if cccl.is_dir():
        args += ["-isystem", str(cccl)]
    if language != "c":
        for part in ("usr/include/c++/12", "usr/include/x86_64-linux-gnu/c++/12", "usr/include/c++/12/backward"):
            args += ["-isystem", str(sysroot / part)]
    for part in ("usr/include/x86_64-linux-gnu", "usr/include"):
        args += ["-isystem", str(sysroot / part)]
    if "mpi" in config["dependencies"]:
        args += ["-isystem", str(sysroot / "usr/lib/x86_64-linux-gnu/openmpi/include")]
    if "openmp" in config["dependencies"]:
        args += ["-isystem", str(sysroot / "usr/lib/llvm-14/lib/clang/14.0.6/include"), "-Xclang", "-fopenmp"]
    for dependency in config["dependencies"]:
        path = environment / dependency / "include"
        if path.is_dir():
            args += ["-I" + str(path)]
        elif dependency in ("eigen", "boost_preprocessor"):
            args += ["-I" + str(environment / dependency)]
    includes = list(config["include_dirs"])
    defines = list(config["defines"])
    extra = list(config["extra_flags"])
    for rule in config["file_rules"]:
        if matches(relative, [rule["match"]]):
            includes += rule.get("include_dirs", [])
            defines += rule.get("defines", [])
            extra += rule.get("extra_flags", [])
    args += ["-I" + str(root / path) for path in includes]
    args += ["-D" + define for define in defines]
    args += extra + ["-x", language]
    return args


def database(root, config, toolchain, environment):
    entries = []
    for source in sources(root, config):
        relative = source.relative_to(root).as_posix()
        cuda = source.suffix == ".cu" or matches(relative, config["cuda_headers"])
        language = "cuda" if cuda else "c" if source.suffix == ".c" else "c++"
        for rule in config["file_rules"]:
            if matches(relative, [rule["match"]]):
                language = rule.get("language", language)
        command = flags(root, config, toolchain, environment, language, relative)
        entries.append({"directory": str(root), "file": str(source), "arguments": command + ["-c", str(source)]})
    return entries


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(content)
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def read_state(output):
    path = output / "state.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def preflight(root, output, activate=True, clion=False):
    state = read_state(output)
    database_path = output / "compile_commands.json"
    if database_path.exists() and not state:
        raise ValueError(f"Unowned database: {database_path}")
    if activate:
        active = root / "build/compile_commands.json"
        owned = active.is_symlink() and active.resolve() == database_path.resolve()
        if (active.exists() or active.is_symlink()) and not owned:
            raise ValueError(f"Preserving existing database: {active}. Use --no-activate, or move it aside yourself.")
    if clion:
        active = root / "compile_commands.json"
        expected = state.get("clion_sha256")
        owned = active.is_file() and not active.is_symlink() and expected and digest(active) == expected
        if (active.exists() or active.is_symlink()) and not owned:
            raise ValueError(f"Preserving existing CLion database: {active}; move it aside yourself.")


def smoke(root, output, config, toolchain, environment):
    """Check actual device-library templates before activating an environment."""
    source = '''#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
namespace {
constexpr int probe_block_threads = 32;  // One CUDA warp exercises CUB reduction.
constexpr int probe_vector_elements = 4;  // Small vector instantiates Thrust storage.
__global__ void probe(int* values) {
  using Reduce = cub::BlockReduce<int, probe_block_threads>;
  __shared__ typename Reduce::TempStorage storage;
  int sum = Reduce(storage).Sum((int)threadIdx.x);
  if (threadIdx.x == 0) { *values = sum; }
}
void host_probe() {
  thrust::device_vector<int> values(probe_vector_elements);
}
}
'''
    if config["toolkit"] == "cuda13":
        # This header's deduction guides exposed the CUDA 13/Clang incompatibility.
        # CUDA 12.4 does not provide this libcu++ header.
        source += '''#include <cuda/std/string_view>
namespace {
constexpr cuda::std::string_view probe_text("cuda");
}
'''
    command = flags(root, config, toolchain, environment, "cuda") + ["-fsyntax-only", "-"]
    result = subprocess.run(command, input=source, text=True, capture_output=True, cwd=root)
    atomic_write(output / "logs/toolchain-smoke.log", result.stdout + result.stderr)
    if result.returncode:
        raise ValueError(f"CUDA/CUB/Thrust syntax probe failed; see {output / 'logs/toolchain-smoke.log'}")


def activate(root, output, entries, toolchain, environment, enable=True, clion=False):
    preflight(root, output, enable, clion)
    state = read_state(output)
    data = json.dumps(entries, indent=2) + "\n"
    path = output / "compile_commands.json"
    atomic_write(path, data)
    if enable:
        active = root / "build/compile_commands.json"
        active.parent.mkdir(parents=True, exist_ok=True)
        if not active.is_symlink():
            active.symlink_to(os.path.relpath(path, active.parent))
    if clion:
        # CLion chooses its project root from the database's real parent directory.
        active = root / "compile_commands.json"
        atomic_write(active, data)
        state["clion_sha256"] = digest(active)
    state.update({"schema": 1, "project": str(root), "clangd": toolchain.clangd,
                  "clangxx": toolchain.clangxx, "environment": str(environment),
                  "database": str(path), "commands": len(entries)})
    atomic_write(output / "state.json", json.dumps(state, indent=2) + "\n")
    print(f"Generated {len(entries)} commands: {path}")
    if clion:
        print(f"CLion: File > Open > {root / 'compile_commands.json'} > Open as Project")


def check(root, output, files):
    root = root.resolve()
    state = read_state(output)
    if not state:
        raise ValueError("Run setup before check")
    entries = {str(Path(entry["file"]).resolve()): entry
               for entry in json.loads(Path(state["database"]).read_text())}
    if not files:
        raise ValueError("Supply files to check or configure check_files")
    failed = []
    for name in files:
        path = (root / name).resolve()
        entry = entries.get(str(path))
        if not entry or not path.is_file():
            raise ValueError(f"File is not in the generated database: {path}")
        prefix = output / "logs" / path.relative_to(root)
        command = entry["arguments"][:-2] + ["-fsyntax-only"]
        header = path.suffix in (".cuh", ".hpp", ".h")
        command += ["-"] if header else [str(path)]
        syntax = subprocess.run(command, cwd=entry["directory"], text=True, capture_output=True,
                                input=f'#include "{path.as_posix()}"\n' if header else None)
        atomic_write(prefix.with_suffix(prefix.suffix + ".syntax.log"), syntax.stdout + syntax.stderr)
        # Isolate checks from inherited .clangd suppressions and remote compiler flags.
        editor = subprocess.run([state["clangd"], "--enable-config=false",
                                 "--compile-commands-dir=" + str(output), "--check=" + str(path),
                                 "--check-lines=1-1"], cwd=root, text=True, capture_output=True)
        atomic_write(prefix.with_suffix(prefix.suffix + ".clangd.log"), editor.stdout + editor.stderr)
        print(f"{name}: compiler {'PASS' if syntax.returncode == 0 else 'FAIL'}; "
              f"clangd {'PASS' if editor.returncode == 0 else 'FAIL'}", flush=True)
        if syntax.returncode or editor.returncode:
            failed.append(name)
    if failed:
        raise ValueError(f"Validation failed for {', '.join(failed)}; logs: {output / 'logs'}")
