pushd src
SET VIDEO_SOURCE=C:\\Users\\yevhe\\PhDProjects\\datasets\\UAV123\\data_seq\\UAV123\\bike2
SET GT_PATH=C:\\Users\\yevhe\\PhDProjects\\datasets\\UAV123\\anno\\UAV123\\bike2.txt
start /B python -m demo --video_path %VIDEO_SOURCE% --gt_path %GT_PATH% --data_type uav123
popd
pause
