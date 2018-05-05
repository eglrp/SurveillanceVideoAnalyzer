#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include "ExtendedViBe.h"

namespace zvs
{

struct RectInfo
{
    cv::Rect rect;
    int matchCount;
    int missCount;

    RectInfo() {};
    RectInfo(const cv::Rect& rectVal) 
        : rect(rectVal), matchCount(0), missCount(0) {}; 
};

class VideoAnalyzer
{
public:
    VideoAnalyzer();
    ~VideoAnalyzer();

    void init(cv::Mat& image);
    void proc(cv::Mat& image);
    int findSplitPosition(int expectCount);
    void release(void);

private:    
    VideoAnalyzer(const VideoAnalyzer& analyzer);
    VideoAnalyzer& operator=(const VideoAnalyzer& analyzer);

    int imageWidth, imageHeight;
    int pixSum;
    int frameCount;
    zsfo::ViBe backModel;
    cv::Mat blurImage, grayImage;
    cv::Mat gradImage, gradForeImage;
    std::vector<RectInfo> saveRects;
    std::vector<cv::Rect> rectsNoUpdate;
    std::vector<double> ratioForeToFull;
    double maxRatioForSparse;
};

}
