pushd src
SET VIDEO_SOURCE=C:\\Users\\yevhe\\PhDProjects\\datasets\\VisDrone\\val\\sequences\\uav0000024_00000_s
SET GT_PATH=C:\\Users\\yevhe\\PhDProjects\\datasets\\VisDrone\\val\\annotations\\uav0000024_00000_s.txt
SET MODEL_CONFIG=siamban.toml
start /B python -m demo --video_path %VIDEO_SOURCE% --model_config %MODEL_CONFIG% --gt_path %GT_PATH% --data_type visdrone
popd
pause
