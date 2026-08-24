from __future__ import annotations

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

try:
    import pybind11
except Exception as exc:  # pragma: no cover - build-time guard
    raise RuntimeError("Building FlexDDM's compiled kernels requires pybind11.") from exc


extra_compile_args = ["/std:c++17"] if __import__("platform").system() == "Windows" and "MSC" in __import__("sys").version else ["-std=c++17", "-O3"]


class OptionalBuildExt(build_ext):
    def run(self) -> None:
        try:
            super().run()
        except Exception as exc:  # pragma: no cover - depends on local toolchain
            print(f"warning: compiled FlexDDM kernels were not built: {exc}")

    def build_extension(self, ext: Extension) -> None:
        try:
            super().build_extension(ext)
        except Exception as exc:  # pragma: no cover - depends on local toolchain
            print(f"warning: skipped {ext.name}: {exc}")

setup(
    ext_modules=[
        Extension(
            "flexddm._cpp",
            ["flexddm/cpp/flexddm_cpp.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": OptionalBuildExt},
)
