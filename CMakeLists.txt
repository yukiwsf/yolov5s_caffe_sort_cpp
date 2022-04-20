cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_FLAGS -std=c++11)

project(demo)

set(OPENCV_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/include)
set(OPENCV_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/lib)
set(CAFFE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/caffe/include)
set(CAFFE_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/caffe/lib)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

aux_source_directory(${SRC_DIR} SRC)

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/samples/track_video.cpp ${SRC} ${CAFFE_INCLUDE_DIR}/caffe/proto/caffe.pb.cc)

# find_package(OpenCV REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${OPENCV_INCLUDE_DIR} ${CAFFE_INCLUDE_DIR} ${INCLUDE_DIR})

# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECT_NAME}  ${OPENCV_LIB_DIR}/libopencv_core.so 
                                       ${OPENCV_LIB_DIR}/libopencv_highgui.so 
                                       ${OPENCV_LIB_DIR}/libopencv_video.so 
                                       ${OPENCV_LIB_DIR}/libopencv_videoio.so 
                                       ${OPENCV_LIB_DIR}/libopencv_imgproc.so 
                                       ${OPENCV_LIB_DIR}/libopencv_imgcodecs.so
                                       ${OPENCV_LIB_DIR}/libopencv_dnn.so
                                       ${CAFFE_LIB_DIR}/libcaffe.so
                                       /usr/lib/x86_64-linux-gnu/libboost_system.so
                                       /usr/lib/x86_64-linux-gnu/libglog.so
                                       /usr/lib/x86_64-linux-gnu/libprotobuf.so)

