call "%~dp0find_python.bat"
pushd src
SET DATASET_ROOT=
SET CLIP=KBrEEi7vJX
SET VIDEO_SOURCE=%DATASET_ROOT%\\%CLIP%.mp4
SET GT_PATH=%DATASET_ROOT%\\%CLIP%_groundtruth.txt
SET MODEL_CONFIG=siamban.toml
start /B %PYTHON_CMD% -m demo --debug --video_path %VIDEO_SOURCE% --model_config %MODEL_CONFIG% --gt_path %GT_PATH% --data_type manual
popd
pause
