# **Finding Lane Lines on the Road**

[gray]: ./examples/gray.jpg "Grayscale"
[gaussian]: ./examples/gaussian.jpg "Gaussian blur"
[canny]: ./examples/canny.jpg "Canny"
[region]: ./examples/region.jpg "Region of interest"
[hough]: ./examples/hough.jpg "Hough lines"

### 1. Requirements

 - Python==3.6.13
 - numpy==1.19.5
 - python-opencv==4.5.2.52
 - matplotlib==3.3.4
 - moviepy==1.0.3


### 2. Pipeline description

In this project, I have created a pipeline able to detect line markings on the road. This pipeline uses a traditional 
openCV approach and its implementation can be found in the P1.ipynb. The line detection is implemented in a dataset 
consisting of 3 videos, which can be viewed after executing the jupyter notebook.

My pipeline consists of the following five steps:

##### 1. Grayscale conversion

![alt text][gray]

##### 2. Gaussian blur

The gaussian blur is implemented with a kernel of size 7.

![alt text][gaussian]

##### 3. Canny edges

The canny edges were implemented with a low threshold of 60 and a high threshold of 180.

![alt text][canny]

##### 4. Region of interest

I selected a trapezoid region of interest that allows to focus only on the area were the lines are. 
I used the following parameters:

 - X bottom left is set to 20% of image width
 - X Bottom right is set to 95% of image width
 - X top left is set to 45% of image width
 - X top right is set to 55% of the image width
 - Y stretches from 100% to 60% of the image height

![alt text][region]

##### 5. Hough lines

The hough lines were implemented with the following parameters:

 - ρ: 1
 - θ: π/180
 - minimum line length: 20
 - maximum line gap: 30

The hough lines algorithm sometimes outputted many lines that were not very relevant. I performed the following steps 
to filter out the most relevant ones, and to merge them into two final lines:

 - I separated the left and right lines based on whether their angle was negative or positive, respectively, and whether
   they appeared on the left or right side of the image.
 - For the left lines I kept those between -33 and -40 degrees and for the right lines those between 25 and 35 degrees.
 - Finally, I kept only the (x_min, x_max) & (y_min, y_max) out of all the left and right lines, and I joined them to 
   make the final lines that are shown in the images. 

![alt text][hough]

### 3. Limitations

A limitation with this approach is that it will not work for left or right lines greater than the limits set inside the 
draw_lines() function. To mitigate that the limits will have to be increased which poses the risk of making the 
algorithm unable to filter correctly the irrelevant lines. The same holds for the region of interest choice.

Another limitation is that the canny edges limits are set so that they work under conditions of relatively high 
contrast. As it becomes easily apparent in the final video, the road conditions can change so that the contrast suddenly
becomes low. This makes the algorithm failing to detect any line on the road, thus potentially confusing a car that may 
rely on these results for its navigation.

This approach assumes only straight lines and is bound to fail whenever there are short turns like on the challenge.mp4 
video. In that case, it will be difficult for a self-driving car to understand that there is a turn with the danger of 
it speeding instead of breaking.

Finally, this pipeline is limited to work only when there are no other cars in its field of view with high contrast 
lines printed on them, triangular traffic signs printed on the road, etc. 


### 4. Ideas for improvement

A possible way to configure the pipeline to better detect turns is to try and detect small segments in the road
lines and join them sequentially, thus creating curvy lines at the end instead of straight ones. 

A way to mitigate the issue of low contrast between the road and the lines would be to modify the parameters of 
the canny edge detection based on the total brightness of the image, thus creating a kind of dynamic pipeline. However, 
this would be an approach very difficult to calibrate for every possible condition.

Finally, to solve the problem posed by other objects appearing inside the region of interest, which confuse the 
algorithm, would be to add an object detection layer which would ignore any detections inside the bounding box of
any object detected on the road.