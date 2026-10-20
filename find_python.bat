:: Picks the interpreter for the run_*.bat scripts and stores it in PYTHON_CMD.
:: Uses uv (project .venv) if installed, otherwise python on PATH
:: (e.g. an activated dtc_training conda env).
:: Lines are prefixed with @ instead of "@echo off" so the caller's echo state is kept.
@where uv >nul 2>nul
@if not errorlevel 1 (
    set "PYTHON_CMD=uv run --project "%~dp0." python"
) else (
    set "PYTHON_CMD=python"
)
@echo Using interpreter: %PYTHON_CMD%
