"""Generate editor integration without replacing unrelated user settings."""

from .environment import ASSETS, digest
from .project import atomic_write


def nvim(config_dir, install=False):
    template = ASSETS / "xedt.lua"
    if install:
        target = config_dir / "lua/xedt.lua"
        stamp = config_dir / ".xedt-module.sha256"
        owned = (target.is_file() and not target.is_symlink() and stamp.is_file()
                 and digest(target) == stamp.read_text().strip())
        if (target.exists() or target.is_symlink()) and not owned:
            raise ValueError(f"Preserving existing Neovim module: {target}")
        if stamp.is_symlink():
            raise ValueError(f"Preserving existing symlink: {stamp}")
        atomic_write(target, template.read_text())
        atomic_write(stamp, digest(target) + "\n")
        print(f"Installed {target}")
    else:
        print(template.read_text())
    print('Load after your existing clangd setup: require("xedt").setup()')
    print("Restart Neovim after integrating the module. CLion uses the root database from setup --clion.")
