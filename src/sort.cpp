#include "sort.h"

Tracker::Tracker() {
    this->trackers.clear();
    this->frame_count = 0;
    std::cout << "\nsort tracker initialize." << std::endl;
}

Tracker::~Tracker() {
    std::cout << "release trakcer sources." << std::endl;
}

/* Computes IoU between two bounding boxes */
double Tracker::GetIoU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt) {
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;
	if (un < DBL_EPSILON)
		return 0;
	return (double)(in / un);
}

/* sort function */
void Tracker::Sort(std::vector<TrackingBox> &detOut, std::vector<TrackingBox> &frameTrackingResult, int imgWidth, int imgHeight) {
	std::cout << "start sort tracking." << std::endl;
    // variables used in each calling sort function
	std::vector< cv::Rect_<float> > predictedBoxes;  // save trk predict result
	std::vector< std::vector<double> > iouMatrix; // save iou with trks and dets
	std::vector<int> assignment;  // save matched det idx for each trk
	std::set<int> unmatchedDetections;  // save unmatched det idx
	std::set<int> unmatchedTrajectories;  // save unmatched trk idx
	std::set<int> allItems;  // save all det idx if det num > trk num
	std::set<int> matchedItems;  // save matched det idx
	std::vector<cv::Point> matchedPairs;  // save matched pair(trk idx, det idx)
	unsigned int trkNum = 0; 
	unsigned int detNum = 0; 
	double total_time = 0.0;
	int64 start_time = 0;
    
	// start calculate fps 
    this->frame_count += 1;
    start_time = cv::getTickCount();

    // if trks is empty
    // initialize trk(kalman tracker) for each detection
    if(this->trackers.size() == 0 && detOut.size() > 0) {
        for(unsigned int i = 0; i < detOut.size(); ++i) {
            KalmanTracker trk(detOut[i].box);
            printf("initialize kalman tracker, id = %d\n", KalmanTracker::kf_count - 1);
            this->trackers.push_back(trk);
        }
        total_time = ((double)(cv::getTickCount() - start_time)) * 1000 / cv::getTickFrequency();
        std::cout << "no trk, dets: " << detOut.size() << std::endl;
        std::cout << "sort once costs: " << total_time << "ms" << std::endl;
        this->frame_count = 0;
        return;
    }
    else if(this->trackers.size() == 0 && detOut.size() == 0) {
        total_time = ((double)(cv::getTickCount() - start_time)) * 1000 / cv::getTickFrequency();
        std::cout << "no trk, no det" << std::endl;
        std::cout << "sort once costs: " << total_time << "ms" << std::endl;
        this->frame_count = 0;
        return;
    }
    
    // predict: predict all trk in trks, save results in predictedBoxes
    predictedBoxes.clear();
    printf("trks is not empty.\n");
    for(auto it = this->trackers.begin(); it != this->trackers.end();) {
        cv::Rect_<float> pBox = it->predict();  
        if(pBox.x > 0 && pBox.y > 0 && pBox.x + pBox.width < imgWidth && pBox.y + pBox.height < imgHeight) {
            predictedBoxes.push_back(pBox);
            ++it;            
        }
        else {
            it = this->trackers.erase(it);
        }
    }

    // if predictedBoxes is empty
    // initialize trk(kalman tracker) for each detection
    if(predictedBoxes.size() == 0 && detOut.size() > 0) {
        for(unsigned int i = 0; i < detOut.size(); ++i) {
            KalmanTracker trk(detOut[i].box);
            printf("initialize kalman tracker, id = %d\n", KalmanTracker::kf_count - 1);
            this->trackers.push_back(trk);
        }
        total_time = ((double)(cv::getTickCount() - start_time)) * 1000 / cv::getTickFrequency();
        std::cout << "no trk, dets: " << detOut.size() << std::endl;
        std::cout << "sort once costs: " << total_time << "ms" << std::endl;
        this->frame_count = 0;
        return;
    }
    else if(predictedBoxes.size() == 0 && detOut.size() == 0) {
        total_time = ((double)(cv::getTickCount() - start_time)) * 1000 / cv::getTickFrequency();
        std::cout << "no trk, no det" << std::endl;
        std::cout << "sort once costs: " << total_time << "ms" << std::endl;
        this->frame_count = 0;
        return;
    }

    // trks has at least 1 element, dets has 0 element
    if(detOut.size() == 0) { 
        printf("dets is empty.\n"); 
        for(auto it = this->trackers.begin(); it != this->trackers.end();) {
            if((it->m_time_since_update < MAX_AGE) && (it->m_hit_streak >= MIN_HITS || this->frame_count <= MIN_HITS)) {
                TrackingBox res;
                res.box = it->get_state();
                res.id = it->m_id;
                res.frame = this->frame_count;
                frameTrackingResult.push_back(res);
                ++it;
            }
            else ++it;        
            // remove dead trk if umatched over MAX_AGE
            if(it != this->trackers.end() && it->m_time_since_update > MAX_AGE) it = this->trackers.erase(it);
        }
        total_time = ((double)(cv::getTickCount() - start_time)) * 1000 / cv::getTickFrequency();
        std::cout << "trks: " << trackers.size() << ", no det" << std::endl;
        std::cout << "sort once costs: " << total_time << "ms" << std::endl;
        return; 
    }

    // assign dets with trks
    trkNum = predictedBoxes.size();
    detNum = detOut.size();
    std::cout << "trkNum = " << trkNum << ", detNum = " << detNum << std::endl;
    /*
                                              det0 det1 det2 det3
    iouMatrix(M*N, rows*cols, trks*dets) = { { x,   x,   x,   x },  trk0
                                             { x,   x,   x,   x },  trk1
                                             { x,   x,   x,   x } } trk2 
    */
    iouMatrix.clear();  
    iouMatrix.resize(trkNum, std::vector<double>(detNum, 0));
    printf("create trks-dets iouMatrix.\n");
    for(unsigned int i = 0; i < trkNum; ++i) {  // compute iou matrix as a distance matrix
        for(unsigned int j = 0; j < detNum; ++j) {
            // use 1 - iou because the hungarian algorithm computes a minimum-cost assignment.
            iouMatrix[i][j] = 1 - GetIoU(predictedBoxes[i], detOut[j].box);
            printf("iouMatrix[trk%d][det%d] = %f\n", i, j, 1 - iouMatrix[i][j]);
        }
    }

    // solve the assignment problem using hungarian algorithm, the resulting assignment is [track(prediction) : detection], with len = preNum
    HungarianAlgorithm HungAlgo;
    assignment.clear();  // every trk label(matched set det idx, umatched set -1) in trks, assignment.size() == trks.size()
    printf("start hungarian matching.\n");
    HungAlgo.Solve(iouMatrix, assignment);
    printf("end hungarian matching.\n");
    for(int i = 0; i < assignment.size(); ++i) {
        if(assignment[i] != -1)
            printf("assignment[trk%d] = det%d, iou = %f\n", i, assignment[i], 1 - iouMatrix[i][assignment[i]]);
    } 

    // find matched trk/det, unmatched trk and unmatched det
    unmatchedTrajectories.clear();
    unmatchedDetections.clear();
    allItems.clear();
    matchedItems.clear();
    std::cout << "trk num = " << trkNum << ", " << "det num = " << detNum  << std::endl;
    for(int i = 0; i < assignment.size(); ++i) {
        if(assignment[i] == -1)  // assignment[i] == -1 means unmatched trk whose idx is i
            unmatchedTrajectories.insert(i);
        else
            matchedItems.insert(assignment[i]);
    }
    for(int i = 0; i < detNum; ++i) {
        std::set<int>::iterator pos = matchedItems.find(i);
        if(pos == matchedItems.end())  // do not find trk i in matched trks/dets
            unmatchedDetections.insert(i);
    }
    std::cout << "before iou thresh: unmatched trk num = " << unmatchedTrajectories.size() << ", unmatched det num = " << unmatchedDetections.size() << ", matched item num = " << matchedItems.size() << std::endl;

    // filter out every matched trk/det in matchedItems by iou threshold, save results in matchedPairs
    matchedPairs.clear();
    for(int i = 0; i < matchedItems.size(); ++i) {
        if (assignment[i] == -1)  // pass over/ignore invalid values
            continue;                
        if(1 - iouMatrix[i][assignment[i]] < IOU_THRESH) {
            unmatchedTrajectories.insert(i);
            unmatchedDetections.insert(assignment[i]);
        }
        else matchedPairs.push_back(cv::Point(i, assignment[i]));  // matchedItems do not update, use matchedPairs instead
    }
    std::cout << "after iou thresh: matched item num = " << matchedPairs.size() << std::endl;

    // update: update each matched trk/det
    int trkIdx, detIdx;
    for(int i = 0; i < matchedPairs.size(); i++) {
        trkIdx = matchedPairs[i].x;
        detIdx = matchedPairs[i].y;
        this->trackers[trkIdx].update(detOut[detIdx].box);
        printf("update trk%d using det%d (x = %f, y = %f, w = %f, h = %f)\n", trkIdx, detIdx, detOut[detIdx].box.x, detOut[detIdx].box.y, detOut[detIdx].box.width, detOut[detIdx].box.height);
    }

    // initialise new trk for each unmatched det
    for(auto umd : unmatchedDetections) {
        KalmanTracker trk(detOut[umd].box);
        printf("initialize kalman tracker, id = %d\n", KalmanTracker::kf_count - 1);
        this->trackers.push_back(trk);
    }

    // get all trk output in trks
    std::cout << "before erase: trakcers.size() = "	<< trackers.size() << std::endl;	
    for(auto it = this->trackers.begin(); it != this->trackers.end();) {
        if((it->m_time_since_update < MAX_AGE) && (it->m_hit_streak >= MIN_HITS || this->frame_count < MIN_HITS)) {
            TrackingBox res;
            res.box = it->get_state();
            res.id = it->m_id;
            res.frame = this->frame_count;
            std::cout << "frame: " << res.frame << " trk result: " << "x = " << res.box.x << ", y = " << res.box.y << ", w = " << res.box.width << ", h = " << res.box.height << ", id = " << res.id << std::endl;
            frameTrackingResult.push_back(res);
            ++it;
        }           
        else if(it->m_time_since_update >= MAX_AGE) 
            it = this->trackers.erase(it);  // remove dead trk if umatched over MAX_AGE
        else ++it; 
    }
    std::cout << "after erase: trakcers.size() = "	<< trackers.size() << std::endl;	

    // end calculate fps
    total_time = ((double)(cv::getTickCount() - start_time)) * 1000 / cv::getTickFrequency();
    std::cout << "sort once costs: " << total_time << "ms" << std::endl;
    std::cout << "end sort tracking." << std::endl;
}