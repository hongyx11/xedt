import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xedt.config import defaults, load
from xedt.environment import Toolchain, archive_for, digest, discover, packages, environment_path
from xedt.editors import nvim
from xedt.project import activate, check, database, preflight


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / ".xedt"
        self.toolchain = Toolchain("clangd", "clang++", self.root / "resources", 17)

    def test_unrelated_database_is_preserved_before_setup(self):
        target = self.root / "build/compile_commands.json"
        target.parent.mkdir()
        target.write_text("user database")
        with self.assertRaisesRegex(ValueError, "Preserving existing"):
            preflight(self.root, self.output)
        self.assertEqual(target.read_text(), "user database")

    def test_broken_unrelated_symlink_is_preserved(self):
        target = self.root / "build/compile_commands.json"
        target.parent.mkdir()
        target.symlink_to("missing-site-database.json")
        with self.assertRaises(ValueError):
            preflight(self.root, self.output)
        self.assertEqual(target.readlink(), Path("missing-site-database.json"))

    def test_repeat_setup_and_edited_clion_copy(self):
        activate(self.root, self.output, [], self.toolchain, self.root / "sdk", clion=True)
        activate(self.root, self.output, [], self.toolchain, self.root / "sdk", clion=True)
        target = self.root / "compile_commands.json"
        target.write_text("independently edited")
        with self.assertRaisesRegex(ValueError, "Preserving existing CLion"):
            preflight(self.root, self.output, clion=True)
        self.assertEqual(target.read_text(), "independently edited")

    def test_corrupt_cache_is_rejected_offline(self):
        original = self.root / "original"
        original.write_bytes(b"valid archive")
        item = {"name": "sample", "url": "https://example.invalid/file.tar", "sha256": digest(original)}
        cache = self.root / "cache"
        (cache / "downloads").mkdir(parents=True)
        (cache / "downloads" / item["sha256"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "No verified cached"):
            archive_for(item, cache, offline=True)

    def test_import_checks_content_instead_of_filename(self):
        reuse = self.root / "reuse"
        reuse.mkdir()
        archive = reuse / "sample"
        archive.write_bytes(b"valid archive")
        item = {"name": "sample", "url": "https://example.invalid/file.tar", "sha256": digest(archive)}
        target = archive_for(item, self.root / "cache", reuse, offline=True)
        self.assertEqual(target.read_bytes(), archive.read_bytes())

    def test_old_clang_is_rejected_for_cuda13(self):
        binary = self.root / "clangd"
        binary.touch()
        with patch("xedt.environment.platform.system", return_value="Darwin"):
            with patch("xedt.environment.version", return_value=17):
                with self.assertRaisesRegex(ValueError, "needs Clang 22"):
                    discover("cuda13", str(binary))

    def test_file_language_and_precision_rules_are_isolated(self):
        for name in ("kernel.cu", "device.cuh", "host.hpp", "main.cpp"):
            (self.root / name).touch()
        config = defaults()
        config["file_rules"] = [{"match": "kernel.cu", "defines": ["VALUE_FP64=1"]}]
        entries = database(self.root, config, self.toolchain, self.root / "sdk")
        commands = {Path(entry["file"]).name: entry["arguments"] for entry in entries}
        self.assertIn("-DVALUE_FP64=1", commands["kernel.cu"])
        self.assertNotIn("-DVALUE_FP64=1", commands["device.cuh"])
        self.assertIn("--cuda-host-only", commands["device.cuh"])
        self.assertNotIn("--cuda-host-only", commands["host.hpp"])
        self.assertNotIn("--cuda-host-only", commands["main.cpp"])
        cuda = commands["kernel.cu"]
        self.assertLess(cuda.index(str(self.toolchain.resource / "include/cuda_wrappers")),
                        cuda.index(str(self.root / "sdk/sysroot/usr/include/c++/12")))

    def test_unknown_config_and_dependency_are_errors(self):
        path = self.root / "settings.json"
        path.write_text('{"typo": true}')
        with self.assertRaisesRegex(ValueError, "Unknown settings"):
            load(path)
        config = defaults()
        config["dependencies"] = ["typo"]
        with self.assertRaisesRegex(ValueError, "Unknown dependency"):
            packages(config)

    def test_compiler_failure_is_not_hidden_by_successful_clangd(self):
        source = self.root / "kernel.cu"
        source.touch()
        entries = [{"directory": str(self.root), "file": str(source),
                    "arguments": ["clang++", "-x", "cuda", "-c", str(source)]}]
        activate(self.root, self.output, entries, self.toolchain, self.root / "sdk")
        from subprocess import CompletedProcess
        with patch("xedt.project.subprocess.run", side_effect=[
            CompletedProcess([], 1, "", "CUDA header error"),
            CompletedProcess([], 0, "", "0 errors"),
        ]):
            with self.assertRaisesRegex(ValueError, "Validation failed"):
                check(self.root, self.output, ["kernel.cu"])
        self.assertIn("CUDA header error", (self.output / "logs/kernel.cu.syntax.log").read_text())

    def test_nvim_installer_preserves_user_edits(self):
        directory = self.root / "nvim"
        nvim(directory, install=True)
        nvim(directory, install=True)
        module = directory / "lua/xedt.lua"
        module.write_text("-- user customization\n")
        with self.assertRaisesRegex(ValueError, "Preserving existing Neovim"):
            nvim(directory, install=True)
        self.assertEqual(module.read_text(), "-- user customization\n")

    def test_fused_preset_selects_verified_baseline_and_project_fmt(self):
        config = defaults("fused-mcl")
        self.assertEqual(config["toolkit"], "cuda12")
        selected = packages(config)
        fmt = next(item for item in selected if item["name"] == "fmt")
        self.assertTrue(fmt["url"].endswith("/11.0.2"))
        self.assertEqual(len(config["check_files"]), 8)

    def test_cuda13_profile_has_complete_locked_components(self):
        config = defaults()
        config["toolkit"] = "cuda13"
        selected = {item["name"]: item for item in packages(config)}
        self.assertIn("cuda_crt", selected)
        self.assertIn("13.2.51", selected["cuda_cudart"]["url"])
        self.assertTrue(selected["cuda_cccl"]["url"].endswith("/v3.4.3/cccl-v3.4.3.tar.gz"))
        self.assertEqual(selected["cuda_cccl"]["sha256"],
                         "8d1f999bd15ca76534420c65039ce8d5c9a638f6e9771225e6ede5ba082c57c1")

    def test_cccl_update_cannot_reuse_broken_cuda13_environment(self):
        config = defaults()
        config["toolkit"] = "cuda13"
        current = packages(config)
        previous = [dict(item) for item in current]
        cccl = next(item for item in previous if item["name"] == "cuda_cccl")
        cccl.update({
            "url": "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-13.2.27-archive.tar.xz",
            "sha256": "56e1bafb29faa87375b0484814870046530b88c0a421909096892f027ec1927b",
        })
        self.assertNotEqual(environment_path(self.root, current), environment_path(self.root, previous))


if __name__ == "__main__":
    unittest.main()
