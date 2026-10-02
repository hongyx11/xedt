# XEDT: Enable CUDA Development on Local MAC OS

A standalone Python package for editing Linux CUDA/C++ projects on macOS with
Neovim or CLion. It downloads checksum-pinned headers, generates a parsing-only
compilation database, and validates both clangd and the native Clang compiler.
No nvcc, CUDA driver, GPU, CMake configuration, or remote connection is needed.

Requires Python 3.9+, Xcode Command Line Tools (`curl`, `ar`, and `tar`), and
native Clang/clangd. Neovim integration uses Neovim 0.11+.

## Install

```sh
git clone https://github.com/hongyx11/xedt.git
cd xedt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
xedt --help
```

For isolated command installation, `pipx install /path/to/xedt` also works.
The package has no runtime Python dependencies. It has not been published to PyPI.

## Set up a project

```sh
xedt init --project /path/to/project
# Review /path/to/project/.xedt.json.
xedt doctor --project /path/to/project
xedt setup --project /path/to/project --clion
xedt check --project /path/to/project src/kernel.cu include/kernel.cuh
```

Default settings select CUDA 12.4.1 and Linux x86-64 GCC 12/glibc headers,
C++20, and `sm_80`. Clang 17+ is accepted for this profile. CUDA 13.2 is an
alternative (`"toolkit": "cuda13"`) paired with CCCL 3.4.3, requiring Clang 22+ and a successful syntax
probe. Compiler version is a preliminary gate; the actual CUDA/CUB/Thrust probe
must pass before the database is activated.

The two named profiles pin specific toolkit releases:

| Profile | Header version | Status on this Mac |
| --- | --- | --- |
| `cuda12` | CUDA 12.4.1 | Verified with Clang/clangd 22.1.8 |
| `cuda13` | CUDA 13.2.0 + CCCL 3.4.3 | Verified with Clang/clangd 22.1.8 |

Select a profile when initializing, or edit `toolkit` in the project settings:

```sh
xedt init --project /path/to/project --toolkit cuda12
xedt init --project /path/to/another-project --toolkit cuda13
```

Both profiles support local editing and navigation with the tested compiler.
Setup requires the syntax probe to pass and reports its diagnostics on failure.
The current accelerator backend is CUDA; HIP, SYCL, and other backends are not implemented.

Discovery considers Homebrew next to Neovim, PATH's Homebrew, common Homebrew
locations, and PATH's clangd. An explicit compiler pair is supported:

```sh
xedt setup --project /path/to/project \
  --clangd /path/to/llvm/bin/clangd --clangxx /path/to/llvm/bin/clang++
```

Edit JSON settings for `source_dirs`, `exclude`, `include_dirs`, `defines`,
`extra_flags`, `standard`, `gpu_arch`, `cuda_headers`, `dependencies`,
`file_rules`, and `check_files`. Paths are relative to the selected project.
Globs use Python fnmatch semantics: `*` may span directories. A `.cuh` defaults
to CUDA; `.hpp` and `.h` default to host C++. Per-file rules can set `language`
to `cuda`, `c++`, or `c` and append flags, defines, or includes.

```json
{
  "schema": 1,
  "toolkit": "cuda12",
  "source_dirs": ["src", "include"],
  "include_dirs": ["include"],
  "dependencies": ["mpi", "fmt"],
  "file_rules": [
    {"match": "include/device/*.hpp", "language": "cuda"},
    {"match": "src/fp64.cu", "defines": ["VALUE_FP64=1"]}
  ],
  "check_files": ["src/fp64.cu", "include/device/product.hpp"]
}
```

Optional pinned packages: `mpi`, `openmp`, `zstd`, `fmt`, `cxxopts`, plus
`cusparse`, `eigen`, `boost_preprocessor`, and `fast_matrix_market` in the CUDA
12 lock. Dependencies must match your project's versions and include layout.
Both profiles pin fmt 11.0.2, matching fused-mcl's declared fallback. The default profile downloads
only the Linux/CUDA core. Optional archives are selected explicitly.

Headers are cached under `~/Library/Caches/xedt/`, separated by the full
selected dependency identity. Archives are verified on every setup run and
unpacked without executing Linux binaries or package scripts. Use `--cache-dir`
to relocate them, `--reuse-downloads DIR` to import existing archives by verified
content, and `--offline` to prohibit downloads.

## Neovim

```sh
xedt nvim --install
```

This installs `~/.config/nvim/lua/xedt.lua`. To use another configuration,
pass `--config-dir DIR`. It preserves independently edited modules and does not
rewrite `init.lua` or an existing LSP configuration. Without `--install`, it
prints the module for inspection.

Load it **after your existing clangd configuration**, then restart Neovim:

```lua
require("xedt").setup()
```

For NvChad with `lua/configs/lspconfig.lua`, add that line after its clangd setup.
Alternatively, an existing config can use `require("xedt").command` as
its `cmd` and associate `.cu`/`.cuh` with `cuda`. The module reads the project's
`.xedt/state.json` when launching the server, so different projects can
use different compiler installations and header environments. Other projects
fall back to `VIM_CLANGD` or PATH's clangd. Existing capabilities and keymaps are
preserved by Neovim's config merge.

Managed CUDA projects launch with `--enable-config=false` and their explicit
database directory. This keeps inherited `.clangd` remote paths, language
overrides, and suppressed errors from changing the generated parsing commands.
Put required compilation flags in `.xedt.json`. The package does not
change `.clangd`; Neovim projects without a package state retain normal config.

## CLion

Run `setup --clion`, then **File > Open > PROJECT/compile_commands.json > Open
as Project**. Keep CLion's bundled language engine. This is a real root-level
copy, because a symlink can cause the IDE to choose the header cache as its
project root. Reload through **Tools > Compilation Database > Reload Compilation
Database Project** after regeneration. See the
[CLion compilation database guide](https://www.jetbrains.com/help/clion/compilation-database.html).

## Existing projects and fused-mcl

```sh
xedt init --project /path/to/fused-mcl --preset fused-mcl
xedt setup --project /path/to/fused-mcl --no-activate --clion
xedt check --project /path/to/fused-mcl
```

The preset uses the CUDA 12.4 parsing baseline and includes MPI, OpenMP, zstd,
fmt, cxxopts, CUDA headers, and FP64
benchmark rules. Settings remain editable; this package does not evaluate CMake
targets. External Ocean/spECK backends require their own includes.

`--no-activate` generates `.xedt/` while preserving an existing
`build/compile_commands.json`. Neovim's module uses the generated database
directly. By default setup activates a relative link at
`build/compile_commands.json`. It refuses to replace a different database.
`--clion` likewise refuses to replace an unrelated root database or one edited
after generation. Review/move existing databases yourself before activating.
Ownership is checked before downloading packages.

You can pass `--config /path/to/settings.json` to use external settings. For
example, `examples/fused-mcl.json` allows verification of this checkout without
adding a settings file to it. Neovim needs a normal root marker such as `.git`
when settings are external.

CUDA 13.2's bundled CCCL has host-only `string_view` deduction guides rejected
by Clang 22. The CUDA 13 profile therefore selects NVIDIA's released CCCL 3.4.3,
which includes the [upstream fix](https://github.com/NVIDIA/cccl/commit/510cc9de1ea656a453d6072ea3e1f6097410db9a).
The [release archive](https://github.com/NVIDIA/cccl/releases/tag/v3.4.3) is verified
against its published SHA-256 and unpacked into the CUDA include prefix.
This uses unmodified upstream headers and keeps compiler diagnostics enabled.
NVIDIA [supports upgrading CCCL separately](https://github.com/NVIDIA/cccl#cuda-toolkit-ctk-compatibility)
from the toolkit. Dependency identity changes select a fresh cached environment,
so a previously extracted incompatible CCCL cannot be reused by accident.
The parsing toolkit does not change the project's actual GPU build environment.

For fused-mcl with CUDA 13, initialize a new project configuration using:

```sh
xedt init --project /path/to/fused-mcl --preset fused-mcl --toolkit cuda13
xedt setup --project /path/to/fused-mcl --no-activate --clion --check
```

If settings already exist, change their `toolkit` to `cuda13`, then rerun setup.

## Verification and limits

`setup` always compiles a CUDA/CUB/Thrust probe before generating the database.
For CUDA 13 it also includes and constructs `cuda::std::string_view` to catch
the deduction-guide regression directly. CUDA 12.4 does not provide that header.
`check` parses each selected real file with both `clang++ -fsyntax-only` and
`clangd --check`. Direct header checks use a tiny including translation unit.
Clangd's `--check-lines=1-1` limits only per-token editor self-tests; the complete
file is parsed. Either compiler or editor failure returns a nonzero status;
logs are saved in `.xedt/logs/` with directory-qualified filenames.

Run `setup --check` to validate configured `check_files` after setup. Rerun setup
after adding/removing files, changing compiler/flags, or moving a checkout.
Generated databases contain absolute paths. Ignore `.xedt/`, root
`compile_commands.json`, and the generated build database in version control;
`.xedt.json` is portable project configuration that may be committed.

These commands support editing, navigation, and parsing. They do not validate
GPU code generation, device linking, runtime behavior, or every build variant.
CLion UI navigation is a separate check from command-line parsing.

Local verification on October 2, 2026: Python 3.9 installation and wheel export;
CUDA 12.4.1 and CUDA 13.2 with CCCL 3.4.3 using Homebrew Clang/clangd 22.1.8;
eight representative fused-mcl files passed both compiler and unsuppressed clangd
checks with each profile; Neovim 0.12.4 attached to clangd and resolved a CUDA
include to its header with each profile. CLion root database generation and
repeat refresh were verified, and its bundled clangd 23 parsed the CUDA 13 engine
header without errors; CLion's UI was not tested. The original CUDA 13.2
CCCL archive failed the syntax probe; the pinned upstream release fixes it.

## Development

```sh
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
python3 -m pip wheel --no-deps . --wheel-dir dist
```

Upstream references: [Neovim LSP](https://neovim.io/doc/user/lsp/),
[clangd configuration](https://clangd.llvm.org/config.html),
[LLVM CUDA support](https://llvm.org/docs/CompileCudaWithLLVM.html),
[CUDA 12.4.1 redistributions](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.4.1.json),
and [CUDA 13.2 redistributions](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.2.0.json).
