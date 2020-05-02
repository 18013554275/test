cmake_minimum_required(VERSION 3.16)
project(test)

include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 14)

#set(OpenCV_DIR "/Users/xysh/opencv/build")
#set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)


add_executable(test main.cpp ImageHandler.cpp ImageHandler.h)

#find_library(OpenCV)
find_package(OpenCV)


target_link_libraries(test ${OpenCV_LIBS})