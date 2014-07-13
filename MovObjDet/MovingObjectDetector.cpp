#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "MovingObjectDetector.h"
#include "VisualInfo.h"
#include "BlobExtractor.h"
#include "BlobTracker.h"
#include "StaticBlobTracker.h"
#include "OperateGeometryTypes.h"
#include "OperateData.h"
#include "CompileControl.h"
#include "RegionOfInterest.h"
#include "CreateDirectory.h"
#include "Timer.h"
#include "FileStreamScopeGuard.h"
#include "Exception.h"

namespace zsfo
{

class MovingObjectDetector::Impl
{
public:
    void init(const StampedImage& input, const std::string& pathPath);
    void init(const StampedImage& input, const cv::Size& normSize, int updateBackInterval,
        int recordSnapshotMode, int saveSnapshotMode, 
        int saveSnapshotInterval, int numOfSnapshotSaved, bool normScale,
        const std::vector<std::vector<cv::Point> >& includeRegionPoints,
        const std::vector<std::vector<cv::Point> >& excludeRegionPoints,
        const std::vector<cv::Point>& recordLoopOrLineSegmentPoints,
        const double* minObjectArea, const double* minObjectWidth, const double* minObjectHeight,
        const bool* charRegionCheck, const std::vector<cv::Rect>& charRegionRects,
        const bool* checkTurnAround, const double* maxDistRectAndBlob,
        const double* minRatioIntersectToSelf, const double* minRatioIntersectToBlob);
    void build(const StampedImage& input);
	void proc(const StampedImage& input, ObjectDetails& output);
	void final(ObjectDetails& output);

private:
    void setConfigParam(bool normScale = true, const double* minObjectArea = 0, 
        const double* minObjectWidth = 0, const double* minObjectHeight = 0,
        const bool* charRegionCheck = 0, const std::vector<cv::Rect>& charRegionRects = std::vector<cv::Rect>(),
        const bool* checkTurnAround = 0, const double* maxDistRectAndBlob = 0,
        const double* minRatioIntersectToSelf = 0, const double* minRatioIntersectToBlob = 0);

    SizeInfo sizeInfo;
    VisualInfo visualInfo;
	BlobExtractor blobExtractor;
	BlobTracker blobTracker;
	StaticBlobTracker staticBlobTracker;
    int updateFullVisualInfoInterval;
	int procCount;
	std::vector<cv::Rect> rects, rectsNoUpdate;	
	cv::Mat initImage, normImage, foreImage, backImage, gradDiffImage;
#if CMPL_CALC_PROC_TIME
    RepeatTimer visualTimer, extractTimer, trackTimer;
#endif
};

}

using namespace std;
using namespace cv;
using namespace ztool;

namespace zsfo
{

void MovingObjectDetector::init(const StampedImage& input, const string& pathPath)
{
    ptrImpl = new Impl;
    ptrImpl->init(input, pathPath);
}

void MovingObjectDetector::init(const StampedImage& input, const Size& normSize, 
    int updateBackInterval, int recordSnapshotMode, int saveSnapshotMode, 
    int saveSnapshotInterval, int numOfSnapshotSaved, bool normScale, 
    const vector<vector<Point> >& includeRegionPoints, 
    const vector<vector<Point> >& excludeRegionPoints, 
    const vector<Point>& recordLoopOrLineSegmentPoints, 
    const double* minObjectArea, const double* minObjectWidth, const double* minObjectHeight,
    const bool* charRegionCheck, const std::vector<cv::Rect>& charRegionRects,
    const bool* checkTurnAround, const double* maxDistRectAndBlob,
    const double* minRatioIntersectToSelf, const double* minRatioIntersectToBlob)
{
    ptrImpl = new Impl;
    ptrImpl->init(input, normSize, updateBackInterval,
        recordSnapshotMode, saveSnapshotMode, saveSnapshotInterval, numOfSnapshotSaved, normScale,
        includeRegionPoints, excludeRegionPoints, recordLoopOrLineSegmentPoints,
        minObjectArea, minObjectWidth, minObjectHeight,
        charRegionCheck, charRegionRects, 
        checkTurnAround, maxDistRectAndBlob, minRatioIntersectToSelf, minRatioIntersectToBlob);
}

void MovingObjectDetector::build(const StampedImage& input)
{
    ptrImpl->build(input);
}

void MovingObjectDetector::proc(const StampedImage& input, ObjectDetails& output)
{
    ptrImpl->proc(input, output);
}

void MovingObjectDetector::final(ObjectDetails& output)
{
    ptrImpl->final(output);
}

void MovingObjectDetector::Impl::init(const StampedImage& input, const string& pathPath)
{
    fstream fileDataSheet;
    FileStreamScopeGuard<fstream> guard(fileDataSheet);
	char stringNotUsed[500];

	// 读取路径
	char pathMOD[500];
	char pathVirtualLoop[500];
	char pathVisualInfo[500];
	char pathBlobExtractor[500];
	char pathBlobTracker[500];
    fileDataSheet.open(pathPath.c_str());
	if (!fileDataSheet.is_open())
	{
        THROW_EXCEPT("cannot open file " + pathPath);
	}
	fileDataSheet >> stringNotUsed >> pathMOD;
	fileDataSheet >> stringNotUsed >> pathVirtualLoop;
	fileDataSheet >> stringNotUsed >> pathVisualInfo;
	fileDataSheet >> stringNotUsed >> pathBlobExtractor;
	fileDataSheet >> stringNotUsed >> pathBlobTracker;
	fileDataSheet.close();
	fileDataSheet.clear();

	// 配置文件
    int recordSnapshotMode;
    int saveScene, saveSlice, saveMask;
    int saveSnapshotInterval, numOfSnapshotSaved;
	int normWidth, normHeight;
	fileDataSheet.open(pathMOD);
	if (!fileDataSheet.is_open())
	{
        THROW_EXCEPT(string("cannot open file ") + pathMOD);
	}
	do
	{
		fileDataSheet >> stringNotUsed;
		if (fileDataSheet.eof())
		{
            THROW_EXCEPT("cannot find config params label [MOD] for MOD");
		}
	}
	while (string(stringNotUsed) != string("[MOD]"));
    fileDataSheet >> stringNotUsed >> recordSnapshotMode;
    fileDataSheet >> stringNotUsed >> saveScene;
    fileDataSheet >> stringNotUsed >> saveSlice;
    fileDataSheet >> stringNotUsed >> saveMask;
    fileDataSheet >> stringNotUsed >> saveSnapshotInterval;
    fileDataSheet >> stringNotUsed >> numOfSnapshotSaved;
	fileDataSheet >> stringNotUsed >> normWidth;
	fileDataSheet >> stringNotUsed >> normHeight;
	fileDataSheet >> stringNotUsed >> updateFullVisualInfoInterval;
	fileDataSheet.close();
	fileDataSheet.clear();

    if (recordSnapshotMode != RecordSnapshotMode::CrossLineSegment &&
        recordSnapshotMode != RecordSnapshotMode::CrossBottomBound &&
        recordSnapshotMode != RecordSnapshotMode::CrossTriBound &&
        recordSnapshotMode != RecordSnapshotMode::Multi &&
        recordSnapshotMode != RecordSnapshotMode::No)
    {
        stringstream message;
        message << "recordSnapshotMode = " << recordSnapshotMode << ", not valid";
        THROW_EXCEPT(message.str());
    }

#if CMPL_WRITE_CONSOLE
	printf("display common config:\n");
	printf("  norm width = %d\n", normWidth);
	printf("  norm height = %d\n", normHeight);
	printf("  full visual info update interval = %d\n", updateFullVisualInfoInterval);
	printf("\n");
#endif

    int saveSnapshotMode = 0;
    if (saveScene)
        saveSnapshotMode += SaveSnapshotMode::SaveScene;
    if (saveSlice)
        saveSnapshotMode += SaveSnapshotMode::SaveSlice;
    if (saveMask)
        saveSnapshotMode += SaveSnapshotMode::SaveMask;

    RegionOfInterest roi;
    VirtualLoop crossLoop;
    LineSegment crossLine;

	// 原始帧
	Mat origFrame = Mat(input.image);
	// 获取归一化图片
	resize(origFrame, initImage, Size(normWidth, normHeight));
	medianBlur(initImage, normImage, 3);
	GaussianBlur(normImage, normImage, Size(3, 3), 0.0);
	// 初始化视觉信息
	visualInfo.init(normImage);
	// 尺寸设置	
	sizeInfo.create(Size(origFrame.cols, origFrame.rows), Size(normWidth, normHeight));
	// 初始化前景提取类
    blobExtractor.init(Size(normWidth, normHeight), pathBlobExtractor, "[BlobExtractor]");
    // 初始化观测区域
    roi.init(Size(normWidth, normHeight), pathVirtualLoop, "[RegionOfInterest]");	
    // 初始化抓拍线圈或者线段 初始化跟踪类
    if (recordSnapshotMode == RecordSnapshotMode::No)
    {
        blobTracker.init(roi, sizeInfo, pathBlobTracker);
    }
    else if (recordSnapshotMode == RecordSnapshotMode::Multi)
    {
        blobTracker.initMultiRecord(roi, sizeInfo, saveSnapshotMode, saveSnapshotInterval, numOfSnapshotSaved, pathBlobTracker);
    }
    else if (recordSnapshotMode == RecordSnapshotMode::CrossLineSegment)
    {
        crossLine.init(pathVirtualLoop, "[CrossLine]");
        blobTracker.initLineSegment(roi, crossLine, sizeInfo, saveSnapshotMode, pathBlobTracker);
    }
    else
    {
        crossLoop.init(pathVirtualLoop, "[SpeedLoop]");
        if (recordSnapshotMode == RecordSnapshotMode::CrossBottomBound)
            blobTracker.initBottomBound(roi, crossLoop, sizeInfo, saveSnapshotMode, pathBlobTracker);
        else if (recordSnapshotMode == RecordSnapshotMode::CrossTriBound)
            blobTracker.initTriBound(roi, crossLoop, sizeInfo, saveSnapshotMode, pathBlobTracker);
    }
	// 静态跟踪
#if CMPL_RUN_STATIC_OBJECT_TRACKER
	staticBlobTracker.init(roi, sizeInfo, pathBlobTracker);
#endif

	procCount = 0;

#if CMPL_SHOW_IMAGE
	Mat temp = Mat::zeros(300, 300, CV_8UC1);
	imshow("temp", temp);
	waitKey(0);
	destroyWindow("temp");
#endif

#if CMPL_WRITE_CONSOLE
	printf("Frame No. %d\n", input.number);
	printf("In MOD_Initialize, time stamp: %lld\n", input.time);
#endif
}

void MovingObjectDetector::Impl::init(const StampedImage& input, const Size& normSize, 
    int updateBackInterval, int recordSnapshotMode, int saveSnapshotMode, 
    int saveSnapshotInterval, int numOfSnapshotSaved, bool normScale,
    const vector<vector<Point> >& includeRegionPoints, 
    const vector<vector<Point> >& excludeRegionPoints, 
    const vector<Point>& recordLoopOrLineSegmentPoints, 
    const double* minObjectArea, const double* minObjectWidth, const double* minObjectHeight,
    const bool* charRegionCheck, const std::vector<cv::Rect>& charRegionRects,
    const bool* checkTurnAround, const double* maxDistRectAndBlob,
    const double* minRatioIntersectToSelf, const double* minRatioIntersectToBlob)
{
    if (normSize.width < 160 || normSize.height < 120)
    {
        stringstream message;
        message << "normSize.width = " << normSize.width << ", "
            << "normSize.height = " << normSize.height << ", "
            << "normSize too small";
        THROW_EXCEPT(message.str());
    }
    if (recordSnapshotMode != RecordSnapshotMode::CrossLineSegment &&
        recordSnapshotMode != RecordSnapshotMode::CrossBottomBound &&
        recordSnapshotMode != RecordSnapshotMode::CrossTriBound &&
        recordSnapshotMode != RecordSnapshotMode::Multi &&
        recordSnapshotMode != RecordSnapshotMode::No)
    {
        stringstream message;
        message << "recordSnapshotMode = " << recordSnapshotMode << ", not valid";
        THROW_EXCEPT(message.str());
    }
    updateFullVisualInfoInterval = updateBackInterval;

    RegionOfInterest roi;
    VirtualLoop crossLoop;
    LineSegment crossLine;

    // 原始帧
	Mat origFrame = Mat(input.image);
    Size origSize = Size(origFrame.cols, origFrame.rows);
	// 归一化图片
	resize(origFrame, initImage, normSize);
	medianBlur(initImage, normImage, 3);
	GaussianBlur(normImage, normImage, Size(3, 3), 0.0);
	// 初始化视觉信息
	visualInfo.init(normImage);
	// 尺寸设置	
	sizeInfo.create(origSize, normSize);
	// 初始化前景提取类
	blobExtractor.init(normSize);
    // 初始化观测和跟踪的感兴趣区域
    vector<vector<Point> > externalPoints;
    // 如果没有指定观测和跟踪的感兴趣区域, 则默认为整个画面
    bool defineIncludedRegion = !includeRegionPoints.empty() || excludeRegionPoints.empty();
    const vector<vector<Point> >& points = defineIncludedRegion ? includeRegionPoints : excludeRegionPoints;
    if (points.empty())
    {
        externalPoints.resize(1);
        externalPoints[0].reserve(4);
        externalPoints[0].push_back(Point(0, 0));
        externalPoints[0].push_back(Point(0, normSize.height));
        externalPoints[0].push_back(Point(normSize.width, normSize.height));
        externalPoints[0].push_back(Point(normSize.width, 0));
    }
    else 
    {
        if (normScale)
            externalPoints = points;
        else
        {
            Size2d scale = div(normSize, origSize);
            int numOfPoly = points.size();
            externalPoints.resize(numOfPoly);
            for (int i = 0; i < numOfPoly; i++)
            {
                int size = points[i].size();
                for (int j = 0; j < size; j++)
                    externalPoints[i][j] = mul(points[i][j], scale);
            }
        }
    }
    roi.init("[RegionOfInterest]", normSize, defineIncludedRegion, externalPoints);
    
    // 初始化抓拍线圈或者线段 初始化跟踪类
    if (recordSnapshotMode == RecordSnapshotMode::No)
    {
        blobTracker.init(roi, sizeInfo);
    }
    else if (recordSnapshotMode == RecordSnapshotMode::Multi)
    {
        blobTracker.initMultiRecord(roi, sizeInfo, saveSnapshotMode, saveSnapshotInterval, numOfSnapshotSaved);
    }
    else if (recordSnapshotMode == RecordSnapshotMode::CrossLineSegment)
    {
        if (recordLoopOrLineSegmentPoints.size() != 2)
        {
            stringstream message;
            message << "recordSnapshotMode == RecordSnapshotMode::CrossLineSegment, "
                << "but recordLoopOrLineSegmentPoints.size() != 2";
            THROW_EXCEPT(message.str());
        }
        vector<Point> internalPoints;
        internalPoints.reserve(2);
        if (normScale)
            internalPoints = recordLoopOrLineSegmentPoints;
        else
        {
            Size2d scale = div(normSize, origSize);
            internalPoints.resize(2);
            internalPoints[0] = mul(recordLoopOrLineSegmentPoints[0], scale);
            internalPoints[1] = mul(recordLoopOrLineSegmentPoints[1], scale);
        }
        crossLine.init(internalPoints[0], internalPoints[1]);
        blobTracker.initLineSegment(roi, crossLine, sizeInfo, saveSnapshotMode);
    }
    else 
    {
        vector<Point> internalPoints;
        internalPoints.reserve(4);
        if (recordLoopOrLineSegmentPoints.empty())
        {
            if (points.empty())
            {
                internalPoints.push_back(Point(5, 5));
                internalPoints.push_back(Point(5, normSize.height - 5));
                internalPoints.push_back(Point(normSize.width - 5, normSize.height - 5));
                internalPoints.push_back(Point(normSize.width - 5, 5));
            }
            else if (defineIncludedRegion && points.size() == 1 && points[0].size() == 4)
            {
                internalPoints = externalPoints[0];
            }
            else
            {
                stringstream strm;
                strm << "recordSnapshotMode == RecordSnapshotMode::CrossBottomBound || "
                    "RecordSnapshotMode::CrossTriBound, "
                    << "includeRegionPoints.size() == " << includeRegionPoints.size() << ", "
                    << "excludeRegionPoints.size() == " << excludeRegionPoints.size() << ", "
                    << "recordLoopOrLineSegmentPoints.size() == 0, "
                    << "unable to determine the actual cross loop points";
                THROW_EXCEPT(strm.str());
            }
        }
        else if (recordLoopOrLineSegmentPoints.size() == 4)
        {
            if (normScale)
                internalPoints = recordLoopOrLineSegmentPoints;
            else
            {
                Size2d scale = div(normSize, origSize);
                internalPoints.resize(4);
                internalPoints[0] = mul(recordLoopOrLineSegmentPoints[0], scale);
                internalPoints[1] = mul(recordLoopOrLineSegmentPoints[1], scale);
                internalPoints[2] = mul(recordLoopOrLineSegmentPoints[2], scale);
                internalPoints[3] = mul(recordLoopOrLineSegmentPoints[3], scale);
            }
        }
        else
        {
            THROW_EXCEPT("recordMode == RecordMode::CrossBottomBoundVisualRecord || RecordMode::CrossTriBoundVisualRecord, "
                "recordLoopOrLineSegmentPoints.size() != 0 && recordLoopOrLineSegmentPoints.size() != 4");
        }
        crossLoop.init("[SpeedLoop]", internalPoints);

        if (recordSnapshotMode == RecordSnapshotMode::CrossBottomBound)
            blobTracker.initBottomBound(roi, crossLoop, sizeInfo, saveSnapshotMode);
        else if (recordSnapshotMode == RecordSnapshotMode::CrossTriBound)
            blobTracker.initTriBound(roi, crossLoop, sizeInfo, saveSnapshotMode);
    }

    // 静态跟踪
#if CMPL_RUN_STATIC_OBJECT_TRACKER
	staticBlobTracker.init(roi, sizeInfo);
#endif

	procCount = 0;

    setConfigParam(normScale, minObjectArea, minObjectWidth, minObjectHeight, charRegionCheck, charRegionRects,
        checkTurnAround, maxDistRectAndBlob, minRatioIntersectToSelf, minRatioIntersectToBlob);

#if CMPL_SHOW_IMAGE
	Mat temp = Mat::zeros(300, 300, CV_8UC1);
	imshow("temp", temp);
	waitKey(0);
	destroyWindow("temp");
#endif

#if CMPL_WRITE_CONSOLE
	printf("Frame No. %d\n", input.number);
	printf("In MOD_Initialize, time stamp: %lld\n", input.time);
#endif
}

void MovingObjectDetector::Impl::setConfigParam(bool normScale, const double* minObjectArea, 
        const double* minObjectWidth, const double* minObjectHeight,
        const bool* charRegionCheck, const vector<Rect>& charRegionRects,
        const bool* checkTurnAround, const double* maxDistRectAndBlob,
        const double* minRatioIntersectToSelf, const double* minRatioIntersectToBlob)
{
    if (normScale)
    {
        blobExtractor.setConfigParams(minObjectArea, minObjectWidth, minObjectHeight, 0,
            charRegionCheck, charRegionRects);
        blobTracker.setConfigParams(checkTurnAround, maxDistRectAndBlob, 
            minRatioIntersectToSelf, minRatioIntersectToBlob);
    }
    else
    {
        double theMinObjectArea, theMinObjectWidth, theMinObjectHeight, theMaxDistRectAndBlob;
        double *ptrMinObjectArea = 0, *ptrMinObjectWidth = 0, *ptrMinObjectHeight = 0, *ptrMaxDistRectAndBlob = 0;
        vector<Rect> rects;
        if (minObjectArea)
        {
            theMinObjectArea = *minObjectArea / sqrt(sizeInfo.horiScale * sizeInfo.vertScale);
            ptrMinObjectArea = &theMinObjectArea;
        }
        if (minObjectWidth)
        {
            theMinObjectWidth = *minObjectWidth / sizeInfo.horiScale;
            ptrMinObjectWidth = &theMinObjectWidth;
        }
        if (minObjectHeight)
        {
            theMinObjectHeight = *minObjectHeight / sizeInfo.vertScale;
            ptrMinObjectHeight = &theMinObjectHeight;
        }
        if (!charRegionRects.empty())
        {
            int size = charRegionRects.size();
            rects.resize(size);
            Size2d scale(1.0 / sizeInfo.horiScale, 1.0 / sizeInfo.vertScale);
            for (int i = 0; i < size; i++)
                rects[i] = mul(charRegionRects[i], scale);
        }
        if (maxDistRectAndBlob)
        {
            theMaxDistRectAndBlob = *maxDistRectAndBlob / sqrt(sizeInfo.horiScale * sizeInfo.vertScale);
            ptrMaxDistRectAndBlob = &theMaxDistRectAndBlob;
        }
        blobExtractor.setConfigParams(ptrMinObjectArea, ptrMinObjectWidth, ptrMinObjectHeight, 0,
            charRegionCheck, rects);
        blobTracker.setConfigParams(checkTurnAround, ptrMaxDistRectAndBlob, 
            minRatioIntersectToSelf, minRatioIntersectToBlob);
    }
}

void MovingObjectDetector::Impl::build(const StampedImage& input)
{
#if CMPL_WRITE_CONSOLE
	printf("Frame No. %d. ", input.number);
	printf("Current time: %lld", input.time);
    printf("\n");
#endif

    // 获取当前原始帧
	Mat origFrame = Mat(input.image);	
	// 计算归一化图片
	resize(origFrame, initImage, Size(sizeInfo.normWidth, sizeInfo.normHeight));
	medianBlur(initImage, normImage, 3);
	GaussianBlur(normImage, normImage, Size(3, 3), 0.0);
#if CMPL_CALC_PROC_TIME
    visualTimer.start();    
#endif
    // 更新视觉信息
	visualInfo.update(normImage, true);
#if CMPL_CALC_PROC_TIME
    visualTimer.end();
#endif
}

void MovingObjectDetector::Impl::proc(const StampedImage& input, ObjectDetails& output)
{
#if CMPL_WRITE_CONSOLE
	printf("Frame No. %d. ", input.number);
	printf("Current time: %lld", input.time);
    printf("\n");
#endif

    // 获取当前原始帧
	Mat origFrame = Mat(input.image);
	// 计算归一化图片
	resize(origFrame, initImage, Size(sizeInfo.normWidth, sizeInfo.normHeight));
	medianBlur(initImage, normImage, 3);
	GaussianBlur(normImage, normImage, Size(3, 3), 0.0);
#if CMPL_CALC_PROC_TIME
    visualTimer.start();    
#endif
    // 更新视觉信息
    visualInfo.update(normImage, foreImage, backImage, gradDiffImage,
        (updateFullVisualInfoInterval == 1) || (procCount++ % updateFullVisualInfoInterval == 0)/*, rectsNoUpdate*/);
#if CMPL_CALC_PROC_TIME
    visualTimer.end();
#endif

#if CMPL_CALC_PROC_TIME
    extractTimer.start();    
#endif
	// 找前景矩形
	blobExtractor.proc(foreImage, normImage, backImage, rects, rectsNoUpdate);
#if CMPL_CALC_PROC_TIME
    extractTimer.end();
#endif

    //if (!rects.empty())
    //{
    //    RepeatTimer timer;
    //    int numRects = rects.size();
    //    for (int i = 0; i < numRects; i++)
    //    {
    //        timer.start();
    //        Scalar r = calcCenterCorrRatio(normImage, backImage, rects[i]);
    //        timer.end();
    //        printf("rect = (%3d, %3d, %3d, %3d): r[0] = %.4f, r[1] = %.4f, r[2] = %.4f,\n", 
    //            rects[i].x, rects[i].y, rects[i].width, rects[i].height, r[0], r[1], r[2]);
    //    }
    //    printf("corr ratio time: %.8f\n", timer.getAccTime());
    //}

#if CMPL_CALC_PROC_TIME
    trackTimer.start();    
#endif
	// 常规目标跟踪和处理
	blobTracker.proc(origFrame, foreImage, input.time, input.number, rects, output.objects);

	// 静态目标跟踪和处理
#if CMPL_RUN_STATIC_OBJECT_TRACKER 
	staticBlobTracker.proc(input.time, input.number, rectsNoUpdate, output.staticObjects);
#endif
#if CMPL_CALC_PROC_TIME
    trackTimer.end();
#endif

#if CMPL_SHOW_IMAGE
    Mat imageForDrawing;
	initImage.copyTo(imageForDrawing);
    // 用白线画出前景矩形
    blobExtractor.drawFinalRects(imageForDrawing, Scalar(255, 255, 255));
    // 画出稳定的矩形区域
	blobExtractor.drawStableRects(imageForDrawing, Scalar(0, 0, 0));
	// 用黄线画出观测和跟踪区域 用红线画出计算速度用虚拟线圈 画出运动目标的矩形和历史
    blobTracker.drawTrackingState(imageForDrawing, 
        Scalar(0, 255, 255), Scalar(0, 0, 255), Scalar(0, 255, 255), Scalar(0, 0, 255));
	// 画静态物体跟踪的矩形
    #if CMPL_RUN_STATIC_OBJECT_TRACKER 
	staticBlobTracker.drawBlobs(imageForDrawing， Scalar(255, 255, 255), Scalar(255, 0, 0));
    #endif
    imshow("Normalized Image With Result", imageForDrawing);
    /*if (!rects.empty())
		waitKey(0);*/
#endif	
}

void MovingObjectDetector::Impl::final(ObjectDetails& output)
{
	blobTracker.final(output.objects);
#if CMPL_CALC_PROC_TIME
    printf("avg visual proc time = %f sec\n", visualTimer.getAvgTime());
    printf("avg extract proc time = %f sec\n", extractTimer.getAvgTime());
    printf("avg track proc time = %f sec\n", trackTimer.getAvgTime());
    printf("\n");
#endif
}

void procVideo(const string& videoName, const string& savePath, 
    const string& sceneName, const string& sliceName, const string& maskName, 
    const string& objectInfoFileName, const string& objectHistoryFileName,
    int procEveryNFrame, const Size& normSize, 
    int buildBackModelCount, int updateBackInterval, 
    int recordSnapshotMode, int saveSnapshotMode, 
    int saveSnapshotInterval, int numOfSnapshotSaved, bool normScale,
    const vector<vector<Point> >& includeRegionPoints,
    const vector<vector<Point> >& excludeRegionPoints,
    const vector<Point>& crossLoopOrLineSegmentPoints,
    const double* minObjectArea, const double* minObjectWidth, const double* minObjectHeight,
    const bool* charRegionCheck, const vector<Rect>& charRegionRects,
    const bool* checkTurnAround, const double* maxDistRectAndBlob,
    const double* minRatioIntersectToSelf, const double* minRatioIntersectToBlob)
{
	if (videoName.empty() || savePath.empty())
    {
        THROW_EXCEPT("videoName and savePath must be assigned");
    }

    if (procEveryNFrame < 1)
    {
        THROW_EXCEPT("procEveryNFrame < 1");
    }

    // 创建保存程序运行结果文件的路径
	createDirectory(savePath);

	VideoCapture inputVideo; 
	Mat frame;
	StampedImage input;
	ObjectDetails output;
	double totalFrameCount;
	int numOfShowProc = 100;
	int showInterval;
	int showCount = 1;

	// 打开视频文件，读取第一帧
    Mutex mtx;
    mtx.lock();
    inputVideo.open(videoName);
    mtx.unlock();
	if (!inputVideo.isOpened())
	{
		THROW_EXCEPT("cannot open file " + videoName);
	}
	input.time = (long long int)inputVideo.get(CV_CAP_PROP_POS_MSEC);
	input.number = (int)inputVideo.get(CV_CAP_PROP_POS_FRAMES);
	totalFrameCount = inputVideo.get(CV_CAP_PROP_FRAME_COUNT);
	showInterval = int(ceil(totalFrameCount) / numOfShowProc);
	inputVideo.read(frame);
	input.image = frame;	
	bool isFrameSmall = input.image.cols <= 1280 && input.image.rows <= 768;

    Timer totalTimer;
    RepeatTimer accProcTimer, accSaveTimer;
	MovingObjectDetector movObjDet;
	// 初始化所有内部变量及参数
	try
	{
        movObjDet.init(input, normSize, updateBackInterval, 
            recordSnapshotMode, saveSnapshotMode, saveSnapshotInterval, numOfSnapshotSaved, 
            normScale, includeRegionPoints, excludeRegionPoints, crossLoopOrLineSegmentPoints,
            minObjectArea, minObjectWidth, minObjectHeight, charRegionCheck, charRegionRects,
            checkTurnAround, maxDistRectAndBlob, minRatioIntersectToSelf, minRatioIntersectToBlob);
	}
	catch (const exception& e)
	{
        THROW_EXCEPT(e.what());
	}

	OutputInfoParser outputParser;
	outputParser.init(savePath, sceneName, sliceName, maskName, 
        objectInfoFileName, objectHistoryFileName, isFrameSmall);

	printf("Process:%7.2f%%\r", 0.0);
    int count = 0;
    int buildCount = 0;
	// 循环处理每一帧
	while (true)
	{
		input.time = (long long int)inputVideo.get(CV_CAP_PROP_POS_MSEC);
		input.number = (int)inputVideo.get(CV_CAP_PROP_POS_FRAMES);
		if (!inputVideo.read(frame))
			break;
        if (++count % procEveryNFrame != 0)
            continue;
		input.image = frame;
		
		// 处理当前传进来的视频帧
#if CMPL_WRITE_CONSOLE
	    Timer currProcTimer;
#endif
		try
		{
            accProcTimer.start();
            if (++buildCount < buildBackModelCount)
                movObjDet.build(input);
            else
                movObjDet.proc(input, output);
            accProcTimer.end();
		}
        catch (const ztool::Exception& e)
        {
            THROW_EXCEPT(e.what());
        }
        catch (const cv::Exception& e)
        {
            THROW_EXCEPT(e.what());
        }
        catch (const std::exception& e)
        {
            THROW_EXCEPT(e.what());
        }

#if CMPL_WRITE_CONSOLE
        printf("Time used in MovingObjectDetector::Proc(): %.4f sec\n", currProcTimer.elapse());
#endif
		if (input.number == showCount * showInterval)
		{
			if (showCount < numOfShowProc)
				printf("Process:%7.2f%%\r", double(showCount) / numOfShowProc  * 100);
			showCount++;
		}
	    // 将信息写入文件
        accSaveTimer.start();
		outputParser.save(output);
        accSaveTimer.end();
		// 将信息展示
#if CMPL_SHOW_IMAGE || CMPL_WRITE_CONSOLE
		outputParser.show(input, output);
#endif
		// 等待控制，便于观察结果
#if CMPL_SHOW_IMAGE
		waitKey(5);
#endif
	}
    mtx.lock();
    inputVideo.release();
    mtx.unlock();
	printf("Process:%7.2f%%\n\n", 100.0);
	// 处理视频结束但是仍然存在于画面中的跟踪对象
	movObjDet.final(output);
	// 将信息写入文件
    accSaveTimer.start();
	outputParser.save(output);
    accSaveTimer.end();
	// 将信息展示
#if CMPL_SHOW_IMAGE || CMPL_WRITE_CONSOLE
	outputParser.show(input, output);
#endif
	outputParser.final();
    totalTimer.end();
    printf("accurate avg proc time = %f\n", accProcTimer.getAvgTime());
    printf("accurate avg save time = %f\n", accSaveTimer.getAvgTime());
    printf("\n");
    printf("total time elapsed = %f\n", totalTimer.elapse());
    printf("avg frame proc time = %f\n", totalTimer.elapse() / count);    
}

}