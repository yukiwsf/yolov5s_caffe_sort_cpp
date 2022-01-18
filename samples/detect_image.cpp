#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "yolo.h"

#define CYCLE_TIMES 1

int main(int argc, char *argv[]) {
    /* init network */
    std::string modelFile = "../model/prototxt/yolov5s_1.prototxt";
    std::string weightFile = "../model/caffemodel/yolov5s_1.caffemodel";
    Detector detector = Detector(modelFile, weightFile);

    /* read/save image path */        
    cv::String imagePath = "../data/test_image.jpg";
    cv::String savePath = "./result.jpg";    

    /* calculate fps */
    double totalTime = 0;
    int totalIterations = 0;

    /* pre-process */
    cv::Mat image = cv::imread(imagePath);
    if(image.empty()) {
        std::cout << "image is empty." << std::endl;
        return -1;
    }
    
    /* start calculate fps */
    float tick = (float)cv::getTickCount();
    
    /* yolo detect */
    std::vector<ObjInfo> objects;
    detector.Detect(image, objects);  

    /* end calculate fps */
    float currentTime = ((float)cv::getTickCount() - tick) / cv::getTickFrequency();
    totalTime += currentTime;
    ++totalIterations;
    if(totalIterations % CYCLE_TIMES == 0) {
        std::cout << "detection time = " << currentTime * 1000 << " ms " << "(mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << std::endl;
    }

    /* draw result */
    for(int i = 0; i < objects.size(); ++i) {
        int x = objects[i].bbox.x;
        int y = objects[i].bbox.y;
        int width = objects[i].bbox.width;
        int height = objects[i].bbox.height;
        cv::String cls = cv::String(clsName[objects[i].clsId]);
        char tmp[8];
        int ret = snprintf(tmp, sizeof(tmp) / sizeof(tmp[0]), "%.2f", objects[i].confidence);
        cv::String conf = tmp;
        cv::String txt = cls + cv::String(" ") + tmp;
        cv::rectangle(image, cv::Rect(x, y, width, height), cv::Scalar(0, 255, 0), 3);
        cv::putText(image, txt, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 3);

    }

    /* write image */
    cv::imwrite(savePath, image);

    /* show result */
    // cv::imshow("out", image);
    // while((char)cv::waitKey(0) == 27) break;

    return 0;
}
