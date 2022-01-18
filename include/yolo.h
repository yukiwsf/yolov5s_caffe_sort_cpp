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

#define YOLO_INPUT_WIDTH 640
#define YOLO_INPUT_HEIGHT 640
#define FPN_STRIDES 3
#define ANCHOR_NUM 3
#define NUM_CLASS 80
#define CONF_THRESHOLD 0.5
#define IS_NMS 1
#define NMS_THRESHOLD 0.5
#define MAX_DET 100

// #define TO_TXT 1

/* class name */
extern const char *clsName[NUM_CLASS];

/* anchor */
extern const std::vector<std::vector<cv::Size2f>> anchors;

/* store object-detection result information */
typedef struct ObjectDetectionInformation {
    int clsId;
    float confidence;
    cv::Rect bbox;
} ObjInfo;

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