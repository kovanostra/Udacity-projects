# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./output_images/test1.jpg)

## 1.Introduction

The goals of this repo are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
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


