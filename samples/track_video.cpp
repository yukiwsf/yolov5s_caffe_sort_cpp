#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "yolo.h"
#include "sort.h"

#define CYCLE_TIMES 1

int main(int argc, char *argv[]) {
    /* init network */
    std::string modelFile = "../model/prototxt/yolov5s_1.prototxt";
    std::string weightFile = "../model/caffemodel/yolov5s_1.caffemodel";
    Detector detector = Detector(modelFile, weightFile);

    /* init video capture and writer */
    cv::VideoCapture inputVideo;          
    cv::String videoInputPath = "../data/test_video.mp4";    
    inputVideo.open(videoInputPath);
    if(!inputVideo.isOpened()) {
        std::cout << "read video failed!" << std::endl;
        return -1;
    }
    int frameNum = inputVideo.get(cv::CAP_PROP_FRAME_COUNT);
    int frameWidth = inputVideo.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter outputVideo;
    cv::String videoOutputPath = "./output_track.avi";
    outputVideo.open(videoOutputPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, cv::Size(frameWidth, frameHeight), true);
        if(!outputVideo.isOpened()) {
        std::cerr << "can not create video file!" << std::endl;
        return -1;
    }

    /* calculate fps */
    double totalTime = 0;
    int totalIterations = 0;

    /* sort params */
    Tracker tracker;
    // only track person
    std::vector<TrackingBox> trackingBoxes;
    std::vector<TrackingBox> trackingResults;

    /* for each frame */
    for(int i = 0; i < frameNum; ++i) {

        /* pre-process */
        cv::Mat image;
        inputVideo >> image;
        if(image.empty()) {
            std::cout << "image is empty." << std::endl;
            continue;
        }
        
        /* start calculate fps */
        float tick = (float)cv::getTickCount();
        
        /* yolo detect */
        std::vector<ObjInfo> objects;
        detector.Detect(image, objects);  
    
        /* sort track */
        for(int j = 0; j < objects.size(); ++j) {
            TrackingBox trackingBox;
            if(objects[j].clsId == 2 || objects[j].clsId == 5 || objects[j].clsId == 7) {  // cls == "car" || "bus" || "truck"
                trackingBox.frame = i;
                trackingBox.id = -1;  // objects[j].clsId;
                trackingBox.box = objects[j].bbox;
                trackingBoxes.push_back(trackingBox);
            }            
        }
        tracker.Sort(trackingBoxes, trackingResults, frameWidth, frameHeight);  // sort 
        std::cout << "tracking results: " << trackingResults.size() << " targets" <<std::endl;

        /* end calculate fps */
        float currentTime = ((float)cv::getTickCount() - tick) / cv::getTickFrequency();
        totalTime += currentTime;
        ++totalIterations;
        if(totalIterations % CYCLE_TIMES == 0) {
            std::cout << "detection time = " << currentTime * 1000 << " ms " << "(mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << std::endl;
        }

        /* draw result */
        for(int k = 0; k < trackingResults.size(); ++k) {
            // std::cout << "frame " << i << ", trk.x = " << trackingResults[k].box.x 
            //                            << ", trk.y = " << trackingResults[k].box.y
            //                            << ", trk.width = " << trackingResults[k].box.width
            //                            << ", trk.height = " << trackingResults[k].box.width << std::endl;
            cv::rectangle(image, 
                            cv::Point(trackingResults[k].box.x, trackingResults[k].box.y), 
                            cv::Point(trackingResults[k].box.x + trackingResults[k].box.width, trackingResults[k].box.y + trackingResults[k].box.height), 
                            cv::Scalar(255, 0, 0), 
                            3);
            char idString[16];
            snprintf(idString, sizeof(idString), "id=%d", trackingResults[k].id);
            cv::putText(image, 
                        cv::String(idString), 
                        cv::Point(trackingResults[k].box.x + trackingResults[k].box.width, trackingResults[k].box.y + trackingResults[k].box.height), 
                        cv::FONT_HERSHEY_SIMPLEX, 
                        1, 
                        cv::Scalar(0, 0, 255), 
                        3);
        }
        
        /* write video */
        outputVideo << image;

        /* clear xxxTrackingBoxes and xxxTrackingResults at the end of each frame */
        trackingResults.clear();
        trackingBoxes.clear();

        /* show result */
        // cv::imshow("out", image);
        // char key = (char)cv::waitKey(10);
        // if(key == 27) break;
    }

    outputVideo.release();
    return 0;
}
