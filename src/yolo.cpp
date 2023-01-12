#include "yolo.h"

bool LetterBox(const cv::Mat &src, cv::Mat &dst, float &d_w, float &d_h, float &ratio) {
    if(src.empty()) {
        std::cout << "LetterBox input image is empty" << std::endl;
        return false;
    }
    int in_w = src.cols;
    int in_h = src.rows;
    int out_w = YOLO_INPUT_WIDTH;
    int out_h = YOLO_INPUT_HEIGHT;
    // choose smaller scale
    float ratio_h = (float)out_h / in_h;
    float ratio_w = (float)out_w / in_w;
    ratio = std::min(ratio_h, ratio_w);
    int resize_w = std::round(in_w * ratio);
    int resize_h = std::round(in_h * ratio);
    float pad_w = out_w - resize_w;
    float pad_h = out_h - resize_h;
    // resize
    cv::resize(src, dst, cv::Size(resize_w, resize_h));
    pad_w /= 2;
    pad_h /= 2;
    // padding gray
    int top = std::round(pad_h - 0.1);
    int bottom = std::round(pad_h + 0.1);
    int left = std::round(pad_w - 0.1);
    int right = std::round(pad_w + 0.1);
    cv::copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    d_w = pad_w;  // d_w = (out_w - in_w * ratio) / 2;
    d_h = pad_h;  // d_h = (out_h - in_h * ratio) / 2;
    return true;
}

/* pre-process: resize, hwc2chw, bgr2rgb, /=255.0 */
void PreProcess(const cv::Mat &src, cv::Mat &dst, float &deltaW, float &deltaH, float &resizeRatio, bool _scaleFill) {
    std::cout << "\nstart yolov5s pre-process." << std::endl;
    if(_scaleFill) {
        LetterBox(src, dst, deltaW, deltaH, resizeRatio);
    }
    else {
        cv::resize(src, dst, cv::Size(YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT));  // resize
    }
    cv::Mat dstCopy = dst.clone();
    int cnt = 0;
    for(int k = 0; k < dst.channels(); ++k) {
        for(int i = 0; i < dst.rows; ++i) {
            for(int j = 0; j < dst.cols; ++j) {            
                int idx = dst.channels() - 1 - k + j * dst.channels() + i * dst.step[0];  // hwc2chw, bgr2rgb
                // int idx = k + j * dst.channels() + i * dst.step[0];  // only hwc2chw
                dstCopy.data[cnt++] = dst.data[idx];
            }
        }
    }
    dstCopy.convertTo(dst, CV_32F, 1.0 / 255, 0);  // /=255.0
    std::cout << "end yolov5s pre-process." << std::endl;
}

/* anchor */
const std::vector< std::vector<cv::Size2f> > anchors = { { { 10,  13 }, { 16,  30  }, { 33,  23  } },
                                                         { { 30,  61 }, { 62,  45  }, { 59,  119 } },
                                                         { { 116, 90 }, { 156, 198 }, { 373, 326 } }, };

/* class name */
const char *clsName[NUM_CLASS] = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                   "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                   "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                   "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                   "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                   "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                   "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                   "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                                   "hair drier", "toothbrush" };

/* sigmoid */
inline float Sigmoid(float x) {
    return 1.0 / (1.0 + std::exp(-x));
}

/* softmax */
void Softmax(std::vector<float> &classes) {
    float sum = 0;
    std::transform(
                   classes.begin(), 
                   classes.end(), 
                   classes.begin(), 
                   [&sum](float score) -> float {
                       float expScore = exp(score);
                       sum += expScore;
                       return expScore;
                   }
                   );
    std::transform(
                   classes.begin(), 
                   classes.end(), 
                   classes.begin(),
                   [sum](float score) -> float { 
                       return score / sum; 
                   }
                   );
}

/* process box in case of crossing the image border*/
cv::Rect BoxBorderProcess(const cv::Rect &box, const int imgWidth, const int imgHeight, float deltaW, float deltaH, bool _scaleFill) {
    int xmin = box.x;
    int ymin = box.y;  
    if(_scaleFill) {
        xmin -= deltaW;
        ymin -= deltaH;
    }
    int xmax = xmin + box.width;
    int ymax = ymin + box.height; 
    xmin = std::max(0, xmin);
    xmin = std::min(imgWidth, xmin);
    ymin = std::max(0, ymin);
    ymin = std::min(imgHeight, ymin);
    xmax = std::max(0, xmax);
    xmax = std::min(imgWidth, xmax);
    ymax = std::max(0, ymax);
    ymax = std::min(imgHeight, ymax);
    int width = xmax - xmin;
    int height = ymax - ymin;  
    return cv::Rect(xmin, ymin, width, height);
};

void DetectLayer(std::vector< caffe::Blob< float >* > &outputs, std::vector<int> &indexes, std::vector<cv::Rect> &boxes, std::vector<cv::Rect> &boxesOffset, std::vector<float> &confidences) {
    /* for every output tensor(stage=0, 1, 2): 1*255*80*80(n, c, h, w), 1*255*40*40, 1*255*20*20 */
    for(int stage = 0; stage < FPN_STRIDES; ++stage) {
        const int downScale = 8 * std::pow(2, stage);  // 8->80*80, 16->40*40, 32->20*20
        caffe::Blob< float >* output = outputs[stage];
        // std::cout << "ouput shape: " << output->shape_string() << std::endl;
        const float *data = (const float*)output->cpu_data();
        /* for every output tensor cell */
        for(int cy = 0; cy < output->height(); ++cy) {
            for(int cx = 0; cx < output->width(); ++cx) {
                for(int na = 0; na < ANCHOR_NUM; ++na) {
                    /* get confidence of every predicted box in each cell */
                    int channel = na * (NUM_CLASS + 5);  // 0 85 170 (255)
                    int confIdx = output->width() * output->height() * (channel + 4) + output->width() * cy + cx;
                    float tc = data[confIdx];
                    float confidence = Sigmoid(tc);
                    /* extract predicted box whose confidence is larger than threshold */
                    if(confidence >= CONF_THRESHOLD) {
                        int xIdx = output->width() * output->height() * (channel + 0) + output->width() * cy + cx;
                        float tx = data[xIdx];
                        int yIdx = output->width() * output->height() * (channel + 1) + output->width() * cy + cx;
                        float ty = data[yIdx];
                        int wIdx = output->width() * output->height() * (channel + 2) + output->width() * cy + cx;
                        float tw = data[wIdx];
                        int hIdx = output->width() * output->height() * (channel + 3) + output->width() * cy + cx;
                        float th = data[hIdx];
                        /* decode xywh from txtytwth */ 
                        float x = (cx + Sigmoid(tx) * 2 - 0.5) * downScale;
                        float y = (cy + Sigmoid(ty) * 2 - 0.5) * downScale;
                        float w = std::pow(Sigmoid(tw) * 2, 2) * anchors[stage][na].width;
                        float h = std::pow(Sigmoid(th) * 2, 2) * anchors[stage][na].height;
                        /* predicted box classification */
                        std::vector<float> classes(NUM_CLASS);
                        for(int i = 0; i < NUM_CLASS; ++i) {
                            int classIdx = output->width() * output->height() * (channel + 5 + i) + output->width() * cy + cx;
                            classes[i] = Sigmoid(data[classIdx]);
                        }
                        // Softmax(classes);
                        auto maxIterator = std::max_element(classes.begin(), classes.end());
                        int maxIndex = (int)(maxIterator - classes.begin());
                        if(NUM_CLASS > 1) {
                            confidence *= classes[maxIndex];
                        }
                        /* predicted box regression */
                        int centerX = (int)x;
                        int centerY = (int)y;
                        int width = (int)w;
                        int height = (int)h;
                        float left = centerX - width / 2;
                        float top = centerY - height / 2;
                        // offset by class
                        float c = maxIndex * 4096;
                        float leftOffset = left + c;
                        float topOffset = top + c;
                        /* store classification, regression and confidence for each predicted box */
                        indexes.push_back(maxIndex);
                        boxes.push_back(cv::Rect((int)left, (int)top, width, height));
                        boxesOffset.push_back(cv::Rect((int)leftOffset, (int)topOffset, (int)width, (int)height));
                        confidences.push_back(confidence);
                    }
                }
            }
        }
    }
}

void PostProcess(std::vector< caffe::Blob< float >* > &outputs, std::vector<ObjInfo> &detResults, int originalImageWidth, int originalImageHeight, float resizeRatio, float deltaW, float deltaH, bool _scaleFill) {
    /* post-process */
    std::cout << "start yolov5s post-process." << std::endl;
    // indexes, boxes and confidences have the same element index
    std::vector<int> indexes;  // vector to store every maxIndex(index of clsName) of predicted box whose confidence is larger than threshold
    std::vector<cv::Rect> boxes;  // vector to store every coordinate(xmin ymin w h) of predicted box whose confidence is larger than threshold
    std::vector<cv::Rect> boxesOffset;
    std::vector<float> confidences;  // vector to store every confidence of predicted box whose confidence is larger than threshold
    DetectLayer(outputs, indexes, boxes, boxesOffset, confidences); 
    /* do nms */
    std::vector<int> indices;  // indices of indexs/boxes/confidences after nms
    std::cout << "-do nms..." << std::endl;
    if(IS_NMS) {
        cv::dnn::NMSBoxes(boxesOffset, confidences, CONF_THRESHOLD, NMS_THRESHOLD, indices);
    } 
    else {
        for(int i = 0; i < boxes.size(); ++i) indices.push_back(i);
    }
    /* limit number of detections */
    if(indices.size() > MAX_DET) indices.resize(MAX_DET);
    /* get yolov5s result */
    std::cout << "-get yolov5s result..." << std::endl;
    float widthScale = (float)YOLO_INPUT_WIDTH / originalImageWidth;  // int divide int will cut-off to int, use int divide float or float divide int or float divide float
    float heightScale = (float)YOLO_INPUT_HEIGHT / originalImageHeight;  // ((float)a) / b
    for(size_t i = 0; i < indices.size(); ++i) {  // for every predicted box
        int idx = indices[i];  // indexes/boxes/confidences index 
        // process every box boundary according to input img size.
        auto newBox = BoxBorderProcess(boxes[idx], YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT, deltaW, deltaH, _scaleFill);  
        ObjInfo object;
        // remap box from input img size to orginal img size
        if(_scaleFill) {
            object.bbox.x = newBox.x / resizeRatio;
            object.bbox.y = newBox.y / resizeRatio;
            object.bbox.width = newBox.width / resizeRatio;
            object.bbox.height = newBox.height / resizeRatio;
        }
        else {
            object.bbox.x = (newBox.x / widthScale);
            object.bbox.y = (newBox.y / heightScale);
            object.bbox.width = (newBox.width / widthScale);
            object.bbox.height = (newBox.height / heightScale);            
        }
        object.confidence = confidences[idx];
        object.clsId = indexes[idx];
        detResults.push_back(object);
        printf("object: %s, xmin = %d, ymin = %d, width = %d, height = %d, conf = %f\n", clsName[object.clsId], object.bbox.x, object.bbox.y, object.bbox.width, object.bbox.height, object.confidence);
    }
    std::cout << "end yolov5s post-process." << std::endl;
}

Detector::Detector(std::string prototxt, std::string caffemodel, bool _scaleFill) {
    /* set device */
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    /* load and init network */
    this->net.reset(new caffe::Net<float>(prototxt, caffe::TEST));
    this->net->CopyTrainedLayersFrom(caffemodel);
    std::cout << "net inputs number is " << this->net->num_inputs() << std::endl;
    std::cout << "net outputs number is " << this->net->num_outputs() << std::endl;
    if(this->net->num_inputs() != 1) 
        std::cerr << "network should have exactly one input." << std::endl;
    this->inputBlob = this->net->input_blobs()[0];
    std::cout << "input data layer channels is " << this->inputBlob->channels() << std::endl;
    std::cout << "input data layer width is " << this->inputBlob->width() << std::endl;
    std::cout << "input data layer height is " << this->inputBlob->height() << std::endl;

    scaleFill = _scaleFill;
    delta_w = 0.0;
    delta_h = 0.0;
    resize_ratio = 1.0;
}

Detector::~Detector() {
    /* release memory */
    this->net.reset();
    std::cout << "release net sources." << std::endl;
}

void Detector::Detect(const cv::Mat &originalImage, std::vector<ObjInfo> &detResults) {
    /* pre-process */
    cv::Mat imagePreprocess; 
    PreProcess(originalImage, imagePreprocess, delta_w, delta_h, resize_ratio, scaleFill);  // hwc2chw, bgr2rgb
    
    /* copy data to input blob */
    memcpy((void*)this->inputBlob->mutable_cpu_data(), (void*)imagePreprocess.data, sizeof(float) * this->inputBlob->count());
    /* clean output blobs */
    this->outputBlobs.clear();     
    /* forward */
    std::cout << "start yolov5s forward." << std::endl;
    this->net->Forward();
    for(int i = 0; i < this->net->num_outputs(); ++i) {
        this->outputBlobs.push_back(this->net->output_blobs()[i]);
    }
    std::cout << "end yolov5s forward." << std::endl;
    #ifdef TO_TXT
    #include <fstream>
    std::ofstream fout("./result.txt");
    for(int i = 0; i < 3; ++i) {
        const float *p = this->outputBlobs[i]->cpu_data();
        size_t num = this->outputBlobs[i]->count();
        printf("\nnum = %ld\n", num);
        for(int j = 0; j < num; ++j) {
            fout << p[j] << " ";
        }
        fout << std::endl;
    }
    fout.close();
    #endif  // #ifdef TO_TXT
    /* post-process */
    int originalImageWidth = originalImage.cols;
    int originalImageHeight = originalImage.rows;
    PostProcess(outputBlobs, detResults, originalImageWidth, originalImageHeight, resize_ratio, delta_w, delta_h, scaleFill);
}
