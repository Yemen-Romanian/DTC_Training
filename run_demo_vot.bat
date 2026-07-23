pushd src
SET VIDEO_SOURCE=C:\\Users\\yevhe\\PhDProjects\\datasets\\TestTrackingDataset\\New\\6\\color
SET GT_PATH=C:\\Users\\yevhe\\PhDProjects\\datasets\\TestTrackingDataset\\New\\6\\groundtruth.txt
SET MODEL_CONFIG=siamban.toml
start /B python -m demo --debug --video_path %VIDEO_SOURCE% --model_config %MODEL_CONFIG% --gt_path %GT_PATH% --data_type vot
popd
pause
