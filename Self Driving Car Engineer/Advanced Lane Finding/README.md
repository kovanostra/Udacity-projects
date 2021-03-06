# Advanced Lane Finding

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](output_images/test1.jpg)

## 1.Introduction

This repo contains code to detect road lanes on either a set of separate images or a single video file.

The goals of this repo are the following:

* Compute the camera-calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## 2. Environment

#### Install the environment

To create a conda environment with the project's dependencies please run the following:

    conda create -n lane-finding python=3.6
    conda activate lane-finding
    pip install requirements.txt

#### Requirements

The repo has been tested to work with the following dependencies:

1. python==3.6.13
1. click==8.0.1
1. matplotlib==3.3.4
1. moviepy==1.0.3
1. numpy==1.19.5
1. opencv-python==4.5.2.52

## 3. Entrypoint

To start the lane detection pipeline first set up the python path

    PYTHONPATH=/path/to/the/repo

For detection on separate images run:

    python src/cli.py detect-images 
    --images_directory /path/to/images/directory/ 
    --calibration_directory /path/to/calibration/images/directory/ 
    --output_directory /directory/to/save/the/output/images 
    --record_all_layers {True} or {False}

Example:

    python src/cli.py detect-images --images_directory test_images --calibration_directory camera_cal --output_directory output_images --record_all_layers False

For detection on a video run:

    python src/cli.py detect-video
    --video_path /path/to/video/file
    --calibration_directory /path/to/calibration/images/directory/ 
    --output_directory /directory/to/save/the/output/images 
    --record_all_layers {True} or {False}

Example:

    python src/cli.py detect-video --video_path project_video.mp4 --calibration_directory camera_cal --output_directory output_video --record_all_layers False

The option record_all_layers will output frames with 3X2 subplots containing:

1. The image gradients
1. The image gradients on the region of interest (ROI)
1. The image gradients with a perspective transform applied
1. The detected lanes polygon
1. The detected polygon on the inverse perspective transform
1. The final frame with the road lanes, the curvature, and the distance from centre

This option is very useful for debugging, but when using it on videos it slows down the pipeline significantly.

## 4. Code structure

    src/
        --domain/
            ---- Contains all the classes which group a common type of operations inside them
        --infrastructure/
            ---- Contains all the project's parameters as constants
        --use_case/
            ---- Contains two use cases and the abstract class they inherit from
        cli.py (The command line interface script)

The repository contains 2 use cases:

1. ImageLanesDetector (responsible for detection on images)
1. VideoLanesDetector (responsible for detection on a video file)

Both are subclasses of the LanesDetector abstract class inside which the pipeline structure is defined. Each use case is
first built and then ran. During the build phase we use the FrameTransformer class to calibrate the camera and then to
obtain the perspective transform matrix (and its inverse).

The pipeline as defined in the _apply_pipeline() method of the LanesDetector class, contains the following steps:

1. Undistort the frame
1. Binarize the frame
1. Isolate the region of interest
1. Apply the perspective transform
1. Detect the road lanes
1. Apply the inverse perspective transform
1. Join lanes and text with the undistorted frame

## 5. Pipeline steps explanation

For a more detailed explanation of the pipeline steps please refer to the following documents:

1. [CAMERA_CALIBRATION.md](documentation/CAMERA_CALIBRATION.md)
1. [PERSPECTIVE_TRANSFORM.md](documentation/PERSPECTIVE_TRANSFORM.md)
1. [IMAGE_BINARIZATION.md](documentation/IMAGE_BINARIZATION.md)
1. [REGION_OF_INTEREST.md](documentation/REGION_OF_INTEREST.md)
1. [LANES_DETECTION.md](documentation/LANES_DETECTION.md)
1. [FINALIZE_FRAME.md](documentation/FINALIZE_FRAME.md)

## 6. Results

#### Frame with all layers drawn

![Lanes Image](output_images/all_layers.jpg)

#### Image results

<table><tr>
<td> <img src="output_images/test1.jpg" alt="Drawing" style="width: 250px;"/> </td>
<td> <img src="output_images/test2.jpg" alt="Drawing" style="width: 250px;"/> </td>
<td> <img src="output_images/test3.jpg" alt="Drawing" style="width: 250px;"/> </td>
</tr></table>
<table><tr>
<td> <img src="output_images/test4.jpg" alt="Drawing" style="width: 250px;"/> </td>
<td> <img src="output_images/test5.jpg" alt="Drawing" style="width: 250px;"/> </td>
<td> <img src="output_images/test6.jpg" alt="Drawing" style="width: 250px;"/> </td>
</tr></table>
<table><tr>
<td> <img src="output_images/straight_lines1.jpg" alt="Drawing" style="width: 250px;"/> </td>
<td> <img src="output_images/straight_lines2.jpg" alt="Drawing" style="width: 250px;"/> </td>
</tr></table>

#### Video results

Please download the videos from the locations below:

[Project video](output_video/project_video.mp4)

[Challenge video](output_video/challenge_video.mp4)

## 7. Discussion

Even though the code on this repo performs reasonably well in most cases tested, there are some limitations to it.

First, the detection time per frame is in the order of 200 ms (plus/minus 50), which makes it not very convenient for
real production. To mitigate that we could implement some optimization steps , such as remove all the drawing parts and
possibly write transfer some or all the code into C++.

Another issue is that sometimes the threshold range selected during image binarization might need to be changed
dynamically based on the lighting conditions and the contrast between the lanes and the road. This can be either due to
shadows or parts of the road being gray.

Moreover, a strong assumption is that there should always be two lanes visible in the road, and they should be one on
the left side of the image and one on the right. In cases where there turns of high curvature the code will not work.

Finally, an important issue arises when objects appear suddenly on top of the lane. In such cases there should be a
complementary object-detection algorithm, able to mask the area of the obstructing objects from the image.