import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

IMPORT_NAME_MAP = {
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
}


TORCH_NAMES = {
    "torch",
    "torchvision",
    "torchaudio",
}


TORCH_SENSITIVE_PACKAGES = {
    "accelerate",
}


def run(cmd):
    print("\nRunning:")
    print(" ".join(str(x) for x in cmd))
    subprocess.check_call(cmd)


def pip_install(*packages, index_url=None, no_deps=False):
    if not packages:
        return

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]

    if index_url:
        cmd += ["--index-url", index_url]

    if no_deps:
        cmd += ["--no-deps"]

    cmd += list(packages)
    run(cmd)


def pip_uninstall(*packages):
    if not packages:
        return

    run([sys.executable, "-m", "pip", "uninstall", "-y", *packages])


def can_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except Exception as e:
        print(f"{module_name} import failed: {type(e).__name__}: {e}")
        return False


def package_base_name(package_spec):
    for sep in ["==", ">=", "<=", "~=", ">", "<"]:
        if sep in package_spec:
            return package_spec.split(sep)[0].strip()
    return package_spec.strip()


def package_exact_version(package_spec):
    if "==" not in package_spec:
        return None
    return package_spec.split("==", 1)[1].strip()


def import_name_for_package(package_spec):
    name = package_base_name(package_spec)
    return IMPORT_NAME_MAP.get(name, name.replace("-", "_"))


def get_site_packages_dir():
    import site

    candidates = []

    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass

    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass

    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.name == "site-packages":
            return path

    for path in map(Path, sys.path):
        if path.exists() and path.name == "site-packages":
            return path

    raise RuntimeError("Could not locate site-packages directory.")


def remove_leftover_torch_files():
    site_packages = get_site_packages_dir()

    patterns = [
        "torch",
        "torch-*.dist-info",
        "torchvision",
        "torchvision-*.dist-info",
        "torchaudio",
        "torchaudio-*.dist-info",
        "functorch",
        "functorch-*.dist-info",
        "~orch",
        "~orch-*",
        "~unctorch",
        "~unctorch-*",
    ]

    print(f"\nCleaning leftover torch files from:\n  {site_packages}")

    for pattern in patterns:
        for path in site_packages.glob(pattern):
            print(f"Removing: {path}")
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


def ensure_bootstrap_packages():
    if not can_import("yaml"):
        print("Installing PyYAML so requirements.yml can be read...")
        pip_install("pyyaml")

    if not can_import("packaging"):
        print("Installing packaging so versions can be checked...")
        pip_install("packaging")


def load_requirements(path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def torch_status():
    try:
        import torch
        from packaging import version

        base = torch.__version__.split("+", 1)[0]

        return {
            "installed": True,
            "version": torch.__version__,
            "base_version": version.parse(base),
            "cuda_build": torch.version.cuda is not None,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "path": getattr(torch, "__file__", None),
        }

    except Exception as e:
        return {
            "installed": False,
            "error": f"{type(e).__name__}: {e}",
        }


def print_torch_status(title):
    status = torch_status()

    print(f"\n{title}:")

    if not status["installed"]:
        print("  installed:      false")
        print(f"  error:          {status['error']}")
        return status

    print(f"  version:        {status['version']}")
    print(f"  cuda build:     {status['cuda_build']}")
    print(f"  cuda version:   {status['cuda_version']}")
    print(f"  cuda available: {status['cuda_available']}")
    print(f"  gpu:            {status['gpu']}")
    print(f"  import path:    {status['path']}")

    return status


def requested_torch_version(torch_config):
    packages = torch_config.get("packages", [])

    for package in packages:
        name = package_base_name(package).lower()
        if name == "torch":
            return package_exact_version(package)

    return None


def versions_match(installed_version, requested_version):
    """
    Examples:
      installed:  2.4.1+cu118
      requested:  2.4.1+cu118  -> True
      requested:  2.4.1        -> True

    This allows requirements.yml to use either:
      torch==2.4.1
    or:
      torch==2.4.1+cu118
    """
    if not requested_version:
        return True

    installed_version = str(installed_version).strip()
    requested_version = str(requested_version).strip()

    if "+" in requested_version:
        return installed_version == requested_version

    return installed_version.split("+", 1)[0] == requested_version


def torch_satisfies_config(status, torch_config):
    if not status["installed"]:
        return False

    from packaging import version

    minimum = version.parse(str(torch_config.get("minimum_version", "0")))
    use_cuda = bool(torch_config.get("cuda", False))

    if status["base_version"] < minimum:
        return False

    if use_cuda and not status["cuda_build"]:
        return False

    if use_cuda and not status["cuda_available"]:
        return False

    requested = requested_torch_version(torch_config)

    if requested and not versions_match(status["version"], requested):
        return False

    return True


def install_torch(torch_config):
    packages = torch_config.get(
        "packages",
        ["torch", "torchvision", "torchaudio"],
    )

    use_cuda = bool(torch_config.get("cuda", False))
    index_url = torch_config.get("cuda_index_url")

    if use_cuda and not index_url:
        raise ValueError("torch.cuda is true, but torch.cuda_index_url is missing.")

    print("\nInstalling torch stack:")
    for package in packages:
        print(f"  - {package}")

    pip_uninstall("torch", "torchvision", "torchaudio")
    remove_leftover_torch_files()

    if use_cuda:
        pip_install(*packages, index_url=index_url)
    else:
        pip_install(*packages)


def ensure_torch(torch_config):
    if not torch_config:
        print("\nNo torch section found. Skipping torch setup.")
        return

    initial_status = print_torch_status("Initial torch status")

    if torch_satisfies_config(initial_status, torch_config):
        print("\nTorch already satisfies requirements. Skipping torch reinstall.")
        return

    print("\nTorch does not satisfy requirements. Reinstalling torch stack...")
    install_torch(torch_config)
    verify_torch(torch_config)


def verify_torch(torch_config):
    if not torch_config:
        print("\nNo torch section found. Skipping torch verification.")
        return

    status = print_torch_status("Torch verification")

    if not torch_satisfies_config(status, torch_config):
        requested = requested_torch_version(torch_config)

        details = [
            "Torch does not satisfy requirements after installation.",
            f"Installed: {status.get('version')}",
            f"Requested torch: {requested}",
            f"CUDA requested: {bool(torch_config.get('cuda', False))}",
            f"CUDA build: {status.get('cuda_build')}",
            f"CUDA available: {status.get('cuda_available')}",
            f"Import path: {status.get('path')}",
        ]

        raise RuntimeError("\n".join(details))

    print("\nTorch verification passed.")


def split_packages(pip_packages):
    normal = []
    torch_sensitive = []
    skipped_torch = []

    for package in pip_packages:
        name = package_base_name(package).lower()

        if name in TORCH_NAMES:
            skipped_torch.append(package)
        elif name in TORCH_SENSITIVE_PACKAGES:
            torch_sensitive.append(package)
        else:
            normal.append(package)

    return normal, torch_sensitive, skipped_torch


def install_regular_packages(pip_packages):
    normal, torch_sensitive, skipped_torch = split_packages(pip_packages)

    if skipped_torch:
        print("\nSkipping torch packages from pip section:")
        for package in skipped_torch:
            print(f"  - {package}")
        print("Torch is managed only by the torch section.")

    if normal:
        print("\nInstalling regular packages:")
        for package in normal:
            print(f"  - {package}")

        pip_install(*normal)
    else:
        print("\nNo regular packages to install.")

    if torch_sensitive:
        print("\nInstalling torch-sensitive packages without dependencies:")
        for package in torch_sensitive:
            print(f"  - {package}")

        print(
            "\nThese are installed with --no-deps so they cannot replace CUDA torch "
            "with a CPU torch wheel."
        )

        pip_install(*torch_sensitive, no_deps=True)


def repo_default_dest(url):
    name = url.rstrip("/").split("/")[-1]

    if name.endswith(".git"):
        name = name[:-4]

    if not name:
        raise ValueError(f"Could not infer repo destination from URL: {url}")

    return Path(name)


def resolve_repo_dest(repo, url, project_dir):
    """
    Resolve where a Git repository should be cloned.

    By default, repos are cloned into:
      <project_dir>/code/<repo-name>

    If repo["dest"] is provided:
      - absolute paths are used as-is
      - relative paths are resolved inside <project_dir>/code
    """
    code_dir = project_dir / "code"
    raw_dest = Path(repo.get("dest") or repo_default_dest(url))

    if raw_dest.is_absolute():
        return raw_dest

    return code_dir / raw_dest


def clone_or_update_repo(repo, project_dir):
    if not isinstance(repo, dict):
        raise TypeError(
            "Each git repo entry must be a mapping, for example: "
            "{url: https://github.com/OWNER/REPO.git, dest: ./repos/REPO}"
        )

    url = repo.get("url")
    if not url:
        raise ValueError("Each git repo entry must include a url.")

    dest = resolve_repo_dest(repo, url, project_dir)
    branch = repo.get("branch")
    tag = repo.get("tag")
    commit = repo.get("commit")
    update = bool(repo.get("update", False))

    if branch and tag:
        raise ValueError(f"Repo cannot specify both branch and tag: {url}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if not (dest / ".git").exists():
            raise RuntimeError(
                f"Destination already exists but is not a git repo: {dest}"
            )

        print(f"\nRepo already exists: {dest}")

        if update:
            print("Updating existing repo...")
            run(["git", "-C", str(dest), "fetch", "--all", "--tags"])

            if branch:
                run(["git", "-C", str(dest), "checkout", branch])
                run(["git", "-C", str(dest), "pull", "--ff-only"])
            elif tag:
                run(["git", "-C", str(dest), "checkout", f"tags/{tag}"])
            elif commit:
                run(["git", "-C", str(dest), "checkout", commit])
            else:
                run(["git", "-C", str(dest), "pull", "--ff-only"])
        else:
            print("Skipping clone. Set update: true to fetch/pull existing repos.")

        return

    cmd = ["git", "clone"]

    if branch:
        cmd += ["--branch", branch]
    elif tag:
        cmd += ["--branch", tag]

    cmd += [url, str(dest)]

    run(cmd)

    if commit:
        run(["git", "-C", str(dest), "checkout", commit])


def clone_git_repos(git_config, project_dir):
    if not git_config:
        print("\nNo git section found. Skipping repo clone.")
        return

    repos = git_config.get("repos", [])

    if not repos:
        print("\nNo git repos listed. Skipping repo clone.")
        return

    if shutil.which("git") is None:
        raise RuntimeError("git is not installed or is not available on PATH.")

    print("\nCloning git repos:")

    for repo in repos:
        url = repo.get("url") if isinstance(repo, dict) else str(repo)
        print(f"  - {url}")
        clone_or_update_repo(repo, project_dir)


def print_environment_info():
    print("\nPython environment:")
    print(f"  executable: {sys.executable}")
    print(f"  version:    {sys.version.split()[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--requirements",
        required=True,
        help="Path to requirements.yml or requirements.yaml",
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help=(
            "Project root directory. Git repos are cloned into "
            "<project-dir>/code by default. If omitted, defaults to the "
            "parent directory of the requirements file's directory."
        ),
    )
    args = parser.parse_args()

    requirements_path = Path(args.requirements)

    if not requirements_path.exists():
        raise FileNotFoundError(
            f"Could not find requirements file: {requirements_path}"
        )

    requirements_path = requirements_path.resolve()

    if args.project_dir:
        project_dir = Path(args.project_dir).resolve()
    else:
        project_dir = requirements_path.parent.parent

    print(f"\nProject directory: {project_dir}")
    print(f"Git repos directory: {project_dir / 'code'}")

    print_environment_info()
    ensure_bootstrap_packages()

    req = load_requirements(requirements_path)

    pip_packages = req.get("pip", [])
    torch_config = req.get("torch", {})
    git_config = req.get("git", {})

    clone_git_repos(git_config, project_dir)

    ensure_torch(torch_config)

    install_regular_packages(pip_packages)

    verify_torch(torch_config)

    print("\nDone.")
    print("Restart Jupyter/kernel/runtime before using the environment.")


if __name__ == "__main__":
    main()
