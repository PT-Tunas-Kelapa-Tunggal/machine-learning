#!/usr/bin/env python3
"""Prepare a standalone, relocatable VariEngine distribution for macOS Apple Silicon (arm64).

Prunes headers, stages krops_vari, the Core ML model, and weather dependencies,
compiles the self-bootstrapping VariPython launcher, relinks Mach-O binaries,
and generates the engine metadata manifest.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run(*args):
    return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True)


def is_macho(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    magic_numbers = {
        b'\xcf\xfa\xed\xfe',  # 64-bit Mach-O
        b'\xce\xfa\xed\xfe',  # 32-bit Mach-O
        b'\xca\xfe\xba\xbe',  # Universal binary
        b'\xca\xfe\xba\xbf',
    }
    try:
        with path.open('rb') as handle:
            return handle.read(4) in magic_numbers
    except Exception:
        return False


LAUNCHER_C_SOURCE = r"""#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <libgen.h>
#include <mach-o/dyld.h>
#include <limits.h>
#include <string.h>

int main(int argc, char *argv[]) {
    char exe_path[PATH_MAX];
    uint32_t size = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &size) != 0) {
        fprintf(stderr, "VariPython: failed to determine executable path\n");
        return 1;
    }
    char *dir = dirname(exe_path);

    char pyhome[PATH_MAX];
    char engine_dir[PATH_MAX];
    // Staging 1: In packaged App bundle (Contents/MacOS/VariPython -> Contents/Resources/VariEngine)
    snprintf(pyhome, sizeof(pyhome), "%s/../Resources/VariEngine/environment", dir);
    snprintf(engine_dir, sizeof(engine_dir), "%s/../Resources/VariEngine", dir);

    // Staging 2: Local staging / CLI execution (VariPython next to VariEngine/)
    if (access(pyhome, F_OK) != 0) {
        snprintf(pyhome, sizeof(pyhome), "%s/VariEngine/environment", dir);
        snprintf(engine_dir, sizeof(engine_dir), "%s/VariEngine", dir);
    }

    char pybin[PATH_MAX];
    snprintf(pybin, sizeof(pybin), "%s/bin/python3", pyhome);
    if (access(pybin, F_OK) != 0) {
        snprintf(pybin, sizeof(pybin), "%s/bin/python3.11", pyhome);
    }

    setenv("PYTHONHOME", pyhome, 1);
    setenv("PYTHONUNBUFFERED", "1", 1);
    setenv("PYTHONDONTWRITEBYTECODE", "1", 1);
    setenv("PYTHONNOUSERSITE", "1", 1);
    setenv("PYTHONTZPATH", "/usr/share/zoneinfo", 0);

    char pypath[PATH_MAX * 2];
    snprintf(pypath, sizeof(pypath), "%s:%s/weather", engine_dir, engine_dir);
    setenv("PYTHONPATH", pypath, 1);

    char model_path[PATH_MAX];
    snprintf(model_path, sizeof(model_path), "%s/models/yolo26m-krops-v0.1.mlpackage", engine_dir);
    setenv("KROPS_MODEL_PATH", model_path, 0);

    char weather_dir[PATH_MAX];
    snprintf(weather_dir, sizeof(weather_dir), "%s/weather", engine_dir);
    setenv("KROPS_WEATHER_DIR", weather_dir, 0);

    char **new_argv = malloc((argc + 1) * sizeof(char *));
    new_argv[0] = pybin;
    for (int i = 1; i < argc; i++) {
        new_argv[i] = argv[i];
    }
    new_argv[argc] = NULL;

    execv(pybin, new_argv);
    perror("VariPython: execv failed");
    return 1;
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True, type=Path, help='Path to Python 3.11 environment prefix')
    parser.add_argument('--package-src', required=True, type=Path, help='Path to integration/krops_vari')
    parser.add_argument('--model-path', required=True, type=Path, help='Path to .mlpackage')
    parser.add_argument('--weather-src', required=True, type=Path, help='Path to weather directory')
    parser.add_argument('--version', required=True, help='Release version tag (e.g. v0.1.0)')
    parser.add_argument('--commit', default='HEAD', help='Git commit SHA')
    parser.add_argument('--output', required=True, type=Path, help='Target staging directory')

    args = parser.parse_args()
    runtime_prefix = args.runtime.resolve()
    package_src = args.package_src.resolve()
    model_path = args.model_path.resolve()
    weather_src = args.weather_src.resolve()
    output_dir = args.output.resolve()

    print(f"=== Preparing VariEngine Distribution ===")
    print(f"Runtime:     {runtime_prefix}")
    print(f"Package:     {package_src}")
    print(f"Model:       {model_path}")
    print(f"Weather:     {weather_src}")
    print(f"Version:     {args.version}")
    print(f"Output:      {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    engine_dir = output_dir / 'VariEngine'
    if engine_dir.exists():
        shutil.rmtree(engine_dir)
    engine_dir.mkdir(parents=True)

    # 1. Copy runtime environment
    print("\n[1/6] Copying Python runtime...")
    dest_env = engine_dir / 'environment'
    run('/bin/cp', '-cR', str(runtime_prefix), str(dest_env))

    # 2. Copy source code and Core ML model
    print("[2/6] Copying krops_vari, model, and weather rules...")
    dest_package = engine_dir / 'krops_vari'
    shutil.copytree(
        package_src,
        dest_package,
        symlinks=True,
        ignore=shutil.ignore_patterns('__pycache__', '.git', '*.pyc', '.venv')
    )

    dest_models = engine_dir / 'models'
    dest_models.mkdir(parents=True, exist_ok=True)
    dest_model = dest_models / model_path.name
    if dest_model.exists():
        shutil.rmtree(dest_model)
    shutil.copytree(model_path, dest_model)

    dest_weather = engine_dir / 'weather'
    dest_weather.mkdir(parents=True, exist_ok=True)
    for rule_file in ('krops_rules.py', 'l6_vari_confidence.py', 'nasa_power_job.py'):
        src_file = weather_src / rule_file
        if src_file.exists():
            shutil.copy2(src_file, dest_weather / rule_file)
            print(f"  Copied weather rule: {rule_file}")

    # Copy plots.json if present
    plots_file = weather_src / 'plots.json'
    if plots_file.exists():
        shutil.copy2(plots_file, dest_weather / 'plots.json')

    # 3. Prune compiler headers, static libs, and build caches
    print("[3/6] Pruning headers, static libs, and caches...")
    for prune_name in ('include', 'lib/cmake', 'lib/pkgconfig', 'share/man', 'share/doc', 'man'):
        prune_path = dest_env / prune_name
        if prune_path.exists():
            shutil.rmtree(prune_path)

    for cache in engine_dir.rglob('__pycache__'):
        shutil.rmtree(cache)

    for static_lib in dest_env.rglob('*.a'):
        static_lib.unlink()

    # 4. Compile the self-bootstrapping native VariPython launcher
    print("[4/6] Compiling native VariPython launcher binary...")
    c_source_path = output_dir / 'VariPython.c'
    c_source_path.write_text(LAUNCHER_C_SOURCE)
    vari_python_bin = output_dir / 'VariPython'
    run('/usr/bin/clang', '-O2', '-arch', 'arm64', '-o', str(vari_python_bin), str(c_source_path))
    c_source_path.unlink()
    os.chmod(vari_python_bin, 0o755)

    # 5. Relink Mach-O binaries to @loader_path and re-sign
    print("[5/6] Relinking Mach-O dynamic libraries and signing...")
    binaries = [p for p in engine_dir.rglob('*') if is_macho(p)]
    print(f"  Found {len(binaries)} Mach-O binaries.")
    for binary in binaries:
        try:
            run('install_name_tool', '-add_rpath', '@loader_path/../lib', str(binary))
        except Exception:
            pass
        try:
            run('codesign', '--force', '--sign', '-', str(binary))
        except Exception:
            pass

    # Sign VariPython with ad-hoc signature
    run('codesign', '--force', '--sign', '-', str(vari_python_bin))

    # 6. Generate engine.json manifest
    print("[6/6] Writing engine.json manifest...")
    manifest = {
        'formatVersion': 1,
        'version': args.version,
        'commit': args.commit,
        'architecture': 'arm64',
        'folder': 'VariEngine',
        'pythonIdentifier': 'com.hata.ControlDroneMapping.engine.vari',
    }
    (engine_dir / 'engine.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print("=== VariEngine preparation complete! ===")


if __name__ == '__main__':
    main()
