# Object tracking evaluation demo app

This is the description of how to run demo application for comparing object tracking results with ground truth. The demo was tested using Python 3.12.7 and Anaconda package manager for installing required libraries.

First, run ``setup_conda_env.bat`` to install the virtual environment and all the dependencies. If you want to just use ``pip`` instead of conda, refer to ``requirements.txt`` file with ``pip install -r requirements.txt`` (but this option was not tested yet).

After installing the dependencies and activating the environment, change directory to ``src`` (``cd src``) and run the demo:
``python -m demo --debug --video_path PATH_TO_VIDEOFILE_OR_IMAGE_FOLDER --gt_path PATH_TO_FILE_WITH_GT --data_type GT_TYPE (synthetic, uav123, manual) --tracker_results PATH_TO_TRACKER_RESULTS_CSV_FILE``,
where:
1. ``PATH_TO_VIDEOFILE_OR_IMAGE_FOLDER`` can be either a path to a video file or folder with images that represent a whole video
2. ``PATH_TO_FILE_WITH_GT`` - path to a ground truth file (labels.txt from Unreal Engine dataset, object_name.txt for UAV123 videos or gt.txt for manually labeled videos, currently only these options are supported).
3. ``--debug`` option, if provided, allows for controlling video frames playback with the ``space`` key. Otherwise, the demo will just process the whole video without pausing.
4. ``--data_type`` is used to understand how to actually parse ground truth file.
5. ``PATH_TO_TRACKER_RESULTS_CSV_FILE`` is a path to .csv file with tracker results. If not provided, the instance of SiamFC tracker will be created, with the weights located in ``output\model_weights\siamfc.pth`` and perform real-time tracking and results visualization. **Note**: for better performace, a GPU should be used for running real-time tracking.

To run a full pipeline, use ``example_files\labels.txt`` and ``example_files\tracking_results.csv`` located in root of the repository with ``--data_type synthetic`` (corresponding Unreal Engine video is called UE5.5_Dataset_2_Dark+Cloudy and is located on Google Drive).

