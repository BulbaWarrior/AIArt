The algorithm will try to approximate a given images with images of anime faces

#The genes#
* The end image will consist of 64 images of anime faces.
* The faces are allowed to be rotated a multiple of 90 degrees (i.e 0, 90, 180 or 270 degrees)
* The gamma of each individual face can also be adjusted by s

#The fitness function#
###Each of the 64 individual small pictures can be evalated separately based on:###
* The everage RGB value of the two pictures (maybe the picture is good, just in a wrong orientation)
* The difference between individual pictures
This can be done in parallel

But also the whole picture should be evaluaded somehow - TODO

#The mutations#
###Mutations are as follows:###
* Change the gamma of each face by some value - always
* Change the orientation of the given image - sometimes
* Change the source image itself - rare

