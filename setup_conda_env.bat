@echo off
:: Conda alternative to "uv sync": builds the dtc_training env from pyproject.toml with pip.
:: torch/torchvision come from the PyTorch cu118 index (only an NVIDIA driver is needed),
:: everything else from PyPI. Versions follow pyproject.toml, not uv.lock, so they may be
:: newer than the locked ones except for the exact pins (torch, torchvision).
call conda create --name dtc_training python=3.12.7 -y
call conda activate dtc_training

:: Split the dependency list into torch packages and the rest.
call python -c "import tomllib, pathlib; d = tomllib.loads(pathlib.Path(r'%~dp0pyproject.toml').read_text()); deps = d['project']['dependencies'] + d['dependency-groups']['dev']; torch = [x for x in deps if x.startswith('torch')]; pathlib.Path(r'%TEMP%\dtc_torch.txt').write_text('\n'.join(torch)); pathlib.Path(r'%TEMP%\dtc_deps.txt').write_text('\n'.join(x for x in deps if x not in torch))"

call python -m pip install -r "%TEMP%\dtc_torch.txt" --index-url https://download.pytorch.org/whl/cu118
call python -m pip install -r "%TEMP%\dtc_deps.txt"
