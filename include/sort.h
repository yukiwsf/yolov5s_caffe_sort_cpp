#ifndef SORT_H
#define SORT_H

#include <string>
#include <set>
#include <vector>
#include <iterator>
#include <algorithm>

#include "kalman_tracker.h"
#include "hungarian.h"

const int MAX_AGE = 1, MIN_HITS = 3;
const float IOU_THRESH = 0.3;

/* store tracking result */
struct TrackingBox {
	int frame;
	int id;
    cv::Rect2d box;
};

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