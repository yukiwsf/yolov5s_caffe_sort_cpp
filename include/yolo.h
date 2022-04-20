#ifndef YOLO_H
#define YOLO_H

#include <string>
#include <vector>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "caffe/caffe.hpp"

constexpr int YOLO_INPUT_WIDTH = 640, YOLO_INPUT_HEIGHT = 640, 
              FPN_STRIDES = 3, ANCHOR_NUM = 3, 
              NUM_CLASS = 80, CONF_THRESHOLD = 0.5, NMS_THRESHOLD = 0.5,
              MAX_DET = 100;
const bool IS_NMS = false; 

// #define TO_TXT 

/* class name */
extern const char *clsName[NUM_CLASS];

/* anchor */
extern const std::vector<std::vector<cv::Size2f>> anchors;

/* store object-detection result information */
struct ObjInfo {
    int clsId;
    float confidence;
    cv::Rect bbox;
};

/* main yolov5s class */
class Detector {
public:
    Detector(std::string prototxt, std::string caffemodel);
    virtual ~Detector();
    void Detect(const cv::Mat &inputImage, std::vector<ObjInfo> &detResults);
private:
    std::shared_ptr< caffe::Net< float > > net;
    caffe::Blob<float> *inputBlob;
    std::vector< caffe::Blob< float >* > outputBlobs;
};

#endif  // #ifndef YOLO_H