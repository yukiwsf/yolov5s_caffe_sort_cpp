#ifndef SORT_H
#define SORT_H

#include <string>
#include <set>
#include <vector>
#include <iterator>
#include <algorithm>

#include "KalmanTracker.h"
#include "Hungarian.h"

#define MAX_AGE 1
#define MIN_HITS 3
#define IOU_THRESH 0.3

/* store tracking result */
typedef struct TrackingBox {
	int frame;
	int id;
    cv::Rect2d box;
} TrackingBox;

/* sort tracking */
class Tracker {
public:
    Tracker();
    virtual ~Tracker();
    double GetIoU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt);
    void Sort(std::vector<TrackingBox> &detOut, std::vector<TrackingBox> &frameTrackingResult,  int imgWidth, int imgHeight);
private:
    int frame_count; 
    std::vector<KalmanTracker> trackers; 
};

#endif  // #ifndef SORT_H