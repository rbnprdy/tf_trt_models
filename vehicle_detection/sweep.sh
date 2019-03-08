#! /bin/bash
python3 video_detection.py parkinglot.mp4 parkinglot_base_640_420.avi 640 420 0
python3 video_detection.py parkinglot.mp4 parkinglot_fast_640_420.avi 640 420 1
python3 video_detection.py parkinglot.mp4 parkinglot_base_1920_1080.avi 1920 1080 0
# python3 video_detection.py parkinglot.mp4 parkinglot_fast_1920_1080.avi 1920 1080 1

