call "%~dp0find_python.bat"
pushd src
SET VIDEO_SOURCE=C:\\Users\\yevhe\\PhDProjects\\datasets\\VOT2018LT_MP4\\car16.mp4
SET MODEL_CONFIG=siamban_efficientnet_fpn_tower_onnx.toml
start /B %PYTHON_CMD% -m demo --video_path %VIDEO_SOURCE% --model_config %MODEL_CONFIG%
popd
pause
