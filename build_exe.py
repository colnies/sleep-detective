"""
Build a standalone Sleep Detective executable and a shareable zip.

    pip install ".[build]"
    python build_exe.py

Produces dist/SleepDetective(.exe) and dist/SleepDetective-<platform>.zip,
which bundles the executable with the sample data and the habit log template.
"""

import os
import platform
import zipfile

import PyInstaller.__main__

from sleep_detective import __version__

NAME = "SleepDetective"
SAMPLE_FILES = [
    "sample_data/fitbit_sleep_data.csv",
    "sample_data/daily_habit_log.csv",
    "sample_data/habit_log_template.csv",
]


def platform_tag() -> str:
    system = {"Windows": "windows", "Darwin": "macos"}.get(platform.system(), platform.system().lower())
    machine = platform.machine().lower()
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine, machine)
    return f"{system}-{arch}"


def main():
    PyInstaller.__main__.run([
        "sleep_detective/__main__.py",
        "--name", NAME,
        "--onefile",
        "--console",
        "--paths", ".",
        "--clean",
        "--noconfirm",
    ])

    exe_name = NAME + (".exe" if platform.system() == "Windows" else "")
    exe_path = os.path.join("dist", exe_name)
    zip_path = os.path.join("dist", f"{NAME}-{__version__}-{platform_tag()}.zip")

    # The executable looks for fitbit_sleep_data.csv and daily_habit_log.csv in its
    # own folder, so the sample data goes right next to it and works out of the box.
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(exe_path, f"{NAME}/{exe_name}")
        for path in SAMPLE_FILES:
            zf.write(path, f"{NAME}/{os.path.basename(path)}")
        zf.write("README.md", f"{NAME}/README.md")

    print(f"\nBuilt {exe_path}\nPackaged {zip_path}")


if __name__ == "__main__":
    main()
