#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "BlobTrackerInternal.h"
#include "FileStreamScopeGuard.h"
#include "Exception.h"
#include "CompileControl.h"

using namespace cv;
using namespace std;
using namespace ztool;

namespace zsfo
{

void BlobTracker::init(const RegionOfInterest& observedRegion, const SizeInfo& sizesOrigAndNorm, const string& path)
{
	ptrImpl = new BlobTrackerImpl;
	ptrImpl->init(observedRegion, sizesOrigAndNorm, path);
}

void BlobTracker::initLineSegment(const RegionOfInterest& observedRegion, const LineSegment& crossLine, 
	const SizeInfo& sizesOrigAndNorm, int saveImageMode, const string& path)
{
	ptrImpl = new BlobTrackerImpl;
	ptrImpl->initLineSegment(observedRegion, crossLine, sizesOrigAndNorm, saveImageMode, path);
}

void BlobTracker::initBottomBound(const RegionOfInterest& observedRegion, const VirtualLoop& catchLoop, 
	const SizeInfo& sizesOrigAndNorm, int saveImageMode, const string& path)
{
	ptrImpl = new BlobTrackerImpl;
	ptrImpl->initBottomBound(observedRegion, catchLoop, sizesOrigAndNorm, saveImageMode, path);
}

void BlobTracker::initTriBound(const RegionOfInterest& observedRegion, const VirtualLoop& catchLoop, 
	const SizeInfo& sizesOrigAndNorm, int saveImageMode, const string& path)
{
	ptrImpl = new BlobTrackerImpl;
	ptrImpl->initTriBound(observedRegion, catchLoop, sizesOrigAndNorm, saveImageMode, path);
}

void BlobTracker::initMultiRecord(const RegionOfInterest& observedRegion, const SizeInfo& sizesOrigAndNorm, 
	int saveImageMode, int saveInterval, int numOfSaved, const string& path)
{
	ptrImpl = new BlobTrackerImpl;
	ptrImpl->initMultiRecord(observedRegion, sizesOrigAndNorm, saveImageMode, saveInterval, numOfSaved, path);
}

void BlobTracker::setConfigParams(const bool* checkTurnAround, const double* maxDistRectAndBlob,
	const double* minRatioIntersectToSelf, const double* minRatioIntersectToBlob)
{
	ptrImpl->setConfigParams(checkTurnAround, maxDistRectAndBlob, 
		minRatioIntersectToSelf, minRatioIntersectToBlob);
}

void BlobTracker::proc(long long int time, int count, const vector<Rect>& rects, vector<ObjectInfo>& objects)
{
	ptrImpl->proc(time, count, rects, objects);
}

void BlobTracker::proc(const Mat& origFrame, const Mat& foreImage, 
	long long int time, int count, const vector<Rect>& rects, vector<ObjectInfo>& objects)
{
	ptrImpl->proc(origFrame, foreImage, time, count, rects, objects);
}

void BlobTracker::proc(const Mat& origFrame, const Mat& foreImage, 
	const Mat& gradDiffImage, const Mat& lastGradDiffImage, 
	long long int time, int count, const vector<Rect>& rects, vector<ObjectInfo>& objects)
{
	ptrImpl->proc(origFrame, foreImage, gradDiffImage, lastGradDiffImage, time, count, rects, objects);
}

void BlobTracker::drawTrackingState(Mat& frame, const cv::Scalar& observedRegionColor, const Scalar& crossLoopOrLineColor,
	const Scalar& blobRectColor, const Scalar& blobHistoryColor) const
{
	ptrImpl->drawTrackingState(frame, observedRegionColor, crossLoopOrLineColor, blobRectColor, blobHistoryColor);
}

void BlobTracker::final(vector<ObjectInfo>& objects) const
{
	ptrImpl->final(objects);
}

void BlobTracker::BlobTrackerImpl::initConfigParam(const string& path)
{
    if (!path.empty())
    {
        fstream initFileStream;
        FileStreamScopeGuard<fstream> guard(initFileStream);
        initFileStream.open(path.c_str());
	    if (!initFileStream.is_open())
	    {
		    //stringstream message;
		    //message << "ERROR in BlobTracker::init(), cannot open file " << path;
		    //throw message.str();
            THROW_EXCEPT("cannot open file " + path);
	    }
	    char stringNotUsed[500];
	    do
	    {
		    initFileStream >> stringNotUsed;
		    if (initFileStream.eof())
		    {
			    //throw string("ERROR in BlobTracker::init(), cannot find config params "
				//             "for label [BlobTracker] BlobTracker\n");
                THROW_EXCEPT("cannot find config params for label [BlobTracker] BlobTracker");
		    }
	    }
	    while(string(stringNotUsed) != string("[BlobTracker]"));

	    initFileStream >> stringNotUsed >> stringNotUsed;
	    initFileStream >> configMatch.runCheckTurnAround;
	    initFileStream >> stringNotUsed;
	    initFileStream >> configMatch.maxDistRectAndBlob;
	    initFileStream >> stringNotUsed;
	    initFileStream >> configMatch.minRatioIntersectToSelf;
	    initFileStream >> stringNotUsed;
	    initFileStream >> configMatch.minRatioIntersectToBlob;
	    initFileStream >> stringNotUsed;
	    initFileStream >> configMatch.maxHistorySizeForDistMatch;
	    initFileStream >> stringNotUsed;
	    initFileStream >> configMatch.maxAvgErrorForDistMatch;
	    initFileStream >> stringNotUsed;
	    initFileStream >> configMatch.runDisplayCalcResults;
	    initFileStream >> stringNotUsed;
	    initFileStream >> configMatch.runShowFitLine;

	    initFileStream.close();
    }
    else
    {
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("BlobTracker is initialized with default param\n");
#endif
        configMatch.runCheckTurnAround = 1;
        configMatch.maxDistRectAndBlob = 15;
        configMatch.minRatioIntersectToSelf = 0.6;
        configMatch.minRatioIntersectToBlob = 0.6;
        configMatch.maxHistorySizeForDistMatch = 0;
        configMatch.maxAvgErrorForDistMatch = 15;
        configMatch.runDisplayCalcResults = false;
        configMatch.runShowFitLine = false;
    }

#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
	printf("display blob tracker config:\n");

	printf("  function match:\n");
	printf("    configMatch.runCheckTurnAround = %s\n", configMatch.runCheckTurnAround ? "true" : "false");
	printf("    configMatch.maxDistRectAndBlob = %.4f\n", configMatch.maxDistRectAndBlob);
	printf("    configMatch.minRatioIntersectToSelf = %.4f\n", configMatch.minRatioIntersectToSelf);
	printf("    configMatch.minRatioIntersectToBlob = %.4f\n", configMatch.minRatioIntersectToBlob);
	printf("    configMatch.maxHistorySizeForDistMatch = %d\n", configMatch.maxHistorySizeForDistMatch);
	printf("    configMatch.maxAvgErrorForDistMatch = %.4f\n", configMatch.maxAvgErrorForDistMatch);
	printf("    configMatch.runDisplayCalcResults = %s\n", configMatch.runDisplayCalcResults ? "true" : "false");
	printf("    configMatch.runShowFitLine = %s\n", configMatch.runShowFitLine ? "true" : "false");

	printf("\n");
#endif
}

void BlobTracker::BlobTrackerImpl::init(const RegionOfInterest& observedRegion, const SizeInfo& sizesOrigAndNorm, const string& path)
{
    roi = new RegionOfInterest(observedRegion);
    sizeInfo = new SizeInfo(sizesOrigAndNorm);
    currTime = new long long int;
    currCount = new int;
    initConfigParam(path);
    blobInstance = new Blob(sizeInfo, currTime, currCount, 0, path);
    blobCount = 0;
}

void BlobTracker::BlobTrackerImpl::initLineSegment(const RegionOfInterest& observedRegion, const LineSegment& crossLine, 
	const SizeInfo& sizesOrigAndNorm, int saveImageMode, const string& path)
{	
    roi = new RegionOfInterest(observedRegion);
    recordLine = new LineSegment(crossLine);
	sizeInfo = new SizeInfo(sizesOrigAndNorm);
    baseRect = new Rect(0, 0, sizeInfo->normWidth, sizeInfo->normHeight);
    currTime = new long long int;
    currCount = new int;
    initConfigParam(path);
    blobInstance = new Blob(recordLine, sizeInfo, baseRect, currTime, currCount, 0, saveImageMode, path);
    blobCount = 0;
}

void BlobTracker::BlobTrackerImpl::initBottomBound(const RegionOfInterest& observedRegion, const VirtualLoop& catchLoop, 
	const SizeInfo& sizesOrigAndNorm, int saveImageMode, const string& path)
{	
    roi = new RegionOfInterest(observedRegion);
    recordLoop = new VirtualLoop(catchLoop);
	sizeInfo = new SizeInfo(sizesOrigAndNorm);
    baseRect = new Rect(0, 0, sizeInfo->normWidth, sizeInfo->normHeight);
    currTime = new long long int;
    currCount = new int;
    initConfigParam(path);
    blobInstance = new Blob(recordLoop, sizeInfo, baseRect, currTime, currCount, 0, false, saveImageMode, path);
    blobCount = 0;
}

void BlobTracker::BlobTrackerImpl::initTriBound(const RegionOfInterest& observedRegion, const VirtualLoop& velocityLoop, 
	const SizeInfo& sizesOrigAndNorm, int saveImageMode, const string& path)
{	
    roi = new RegionOfInterest(observedRegion);
    recordLoop = new VirtualLoop(velocityLoop);
	sizeInfo = new SizeInfo(sizesOrigAndNorm);
    baseRect = new Rect(0, 0, sizeInfo->normWidth, sizeInfo->normHeight);
    currTime = new long long int;
    currCount = new int;
    initConfigParam(path);
    blobInstance = new Blob(recordLoop, sizeInfo, baseRect, currTime, currCount, 0, true, saveImageMode, path);
    blobCount = 0;
}

void BlobTracker::BlobTrackerImpl::initMultiRecord(const RegionOfInterest& observedRegion, const SizeInfo& sizesOrigAndNorm, 
    int saveImageMode, int saveInterval, int numOfSaved, const string& path)
{
    roi = new RegionOfInterest(observedRegion);
	sizeInfo = new SizeInfo(sizesOrigAndNorm);
    baseRect = new Rect(0, 0, sizeInfo->normWidth, sizeInfo->normHeight);
    currTime = new long long int;
    currCount = new int;
    initConfigParam(path);
    blobInstance = new Blob(sizeInfo, baseRect, currTime, currCount, 0, saveImageMode, saveInterval, numOfSaved, path);
    blobCount = 0;
}

void BlobTracker::BlobTrackerImpl::setConfigParams(const bool* checkTurnAround, const double* maxDistRectAndBlob,
    const double* minRatioIntersectToSelf, const double* minRatioIntersectToBlob)
{
    if (!(checkTurnAround || maxDistRectAndBlob || minRatioIntersectToSelf || minRatioIntersectToBlob))
        return;

#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
    printf("Some config param(s) of BlobTracker set:\n");
#endif
    if (checkTurnAround)
    {
        configMatch.runCheckTurnAround = *checkTurnAround;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("  configMatch.runCheckTurnAround = %s\n", configMatch.runCheckTurnAround ? "true" : "false");
#endif
    }
    if (maxDistRectAndBlob)
    {
        configMatch.maxDistRectAndBlob = *maxDistRectAndBlob;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("  configMatch.maxDistRectAndBlob = %.4f\n", configMatch.maxDistRectAndBlob);
#endif
    }
    if (minRatioIntersectToSelf)
    {
        configMatch.minRatioIntersectToSelf = *minRatioIntersectToSelf;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("  configMatch.minRatioIntersectToSelf = %.4f\n", configMatch.minRatioIntersectToSelf);
#endif
    }
    if (minRatioIntersectToBlob)
    {
        configMatch.minRatioIntersectToBlob = *minRatioIntersectToBlob;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("  configMatch.minRatioIntersectToBlob = %.4f\n", configMatch.minRatioIntersectToBlob);
#endif
    }
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
    printf("\n");
#endif
}

void BlobTracker::BlobTrackerImpl::proc(long long int time, int count, 
	const vector<Rect>& rects, vector<ObjectInfo>& objects)
{
	// 设置公共变量
	*currTime = time;
	*currCount = count;
	// 第一次更新 blobTracker 结构体中的 blobList 链表，里面记录了上一帧中存在的运动目标
    // 如果检测到 blobList 中没有记录的前景，则将这个前景 push 到 blobList 链表中，成为新的运动目标
    // 对于要 blobList 中即将删除的运动目标进行标记，在后面的 updateBlobListAfterCheck 函数中才进行删除
	updateBlobListBeforeCheck(rects);
	// 抓拍图片，判断是否违章
	updateState();
	// 处于跟踪状态的目标输出即时信息，跟踪结束的目标输出完整信息
	outputInfo(objects, false);
	// 第二次更新 blobTracker 结构体中的 blobList 链表
	updateBlobListAfterCheck();
}

void BlobTracker::BlobTrackerImpl::proc(const Mat& origFrame, const Mat& foreImage, 
    long long int time, int count, const vector<Rect>& rects, vector<ObjectInfo>& objects)
{
	// 设置公共变量
	*currTime = time;
	*currCount = count;
	// 第一次更新 blobTracker 结构体中的 blobList 链表，里面记录了上一帧中存在的运动目标
    // 如果检测到 blobList 中没有记录的前景，则将这个前景 push 到 blobList 链表中，成为新的运动目标
    // 对于要 blobList 中即将删除的运动目标进行标记，在后面的 updateBlobListAfterCheck 函数中才进行删除
	updateBlobListBeforeCheck(rects);
	// 抓拍图片，判断是否违章
	updateState(origFrame, foreImage);
	// 处于跟踪状态的目标输出即时信息，跟踪结束的目标输出完整信息
	outputInfo(objects, false);
	// 第二次更新 blobTracker 结构体中的 blobList 链表
	updateBlobListAfterCheck();
}

void BlobTracker::BlobTrackerImpl::proc(const Mat& origFrame, const Mat& foreImage, 
	const Mat& gradDiffImage, const Mat& lastGradDiffImage, long long int time, int count, 
	const vector<Rect>& rects, vector<ObjectInfo>& objects)
{
	// 设置公共变量
	*currTime = time;
	*currCount = count;
	// 第一次更新 blobTracker 结构体中的 blobList 链表，里面记录了上一帧中存在的运动目标
    // 如果检测到 blobList 中没有记录的前景，则将这个前景 push 到 blobList 链表中，成为新的运动目标
    // 对于要 blobList 中即将删除的运动目标进行标记，在后面的 updateBlobListAfterCheck 函数中才进行删除
	updateBlobListBeforeCheck(rects);
	// 抓拍图片，判断是否违章
	updateState(origFrame, foreImage, gradDiffImage, lastGradDiffImage);
	// 处于跟踪状态的目标输出即时信息，跟踪结束的目标输出完整信息
	outputInfo(objects, false);
	// 第二次更新 blobTracker 结构体中的 blobList 链表
	updateBlobListAfterCheck();
}

void BlobTracker::BlobTrackerImpl::final(vector<ObjectInfo>& objects) const
{
    outputInfo(objects, true);
}

void BlobTracker::BlobTrackerImpl::addBlob(const Rect& rect)
{
	++blobCount;
    Blob* ptrBlob = blobInstance->createNew(blobCount, rect)/*new Blob(*blobInstance, blobCount, rect)*/;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
	printf("Blob ID: %d Begin tracking. Time stamp: %lld, Frame count: %d\n", blobCount, *currTime, *currCount);
#endif
	blobList.push_back(ptrBlob);    
	if (blobCount >= 1000000)
		blobCount = 0;
}

void BlobTracker::BlobTrackerImpl::updateBlobListBeforeCheck(const vector<Rect>& rects)
{
	match(rects);
}

}

namespace
{

void linearRegres(const vector<Point>& history, Point& pointInLine, Point2d& dirVector, double& avgError)
{
	double N = history.size();	
	double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
	for (int i = 0; i < N; i++)
	{
		sumX += history[i].x;
		sumY += history[i].y;
		sumXY += history[i].x * history[i].y;
		sumX2 += history[i].x * history[i].x;
		sumY2 += history[i].y * history[i].y;
	}

	double dX2 = N * sumX2 - sumX * sumX;
	double dY2 = N * sumY2 - sumY * sumY;
	double dXY = N * sumXY - sumX * sumY;

	pointInLine.x = sumX / N;
	pointInLine.y = sumY / N;

	double theta = atan2(2 * dXY, dX2 - dY2) / 2;
	dirVector.x = cos(theta);
	dirVector.y = sin(theta);

	double sumError = 0;
	for (int i = 0; i < N; i++)
	{
		sumError += fabs(dirVector.y * history[i].x + dirVector.x * pointInLine.y -
			             dirVector.y * pointInLine.x - dirVector.x * history[i].y);
	}
	avgError = sumError / N;
}

}

namespace zsfo
{

void BlobTracker::BlobTrackerImpl::match(const vector<Rect>& rects)
{
	// 如果 blobList 为空，rects 为空，不进行任何操作，直接返回
	if (blobList.empty() && rects.empty())
		return;

	// 如果 blobList 为空，rects 不为空，则添加新的跟踪对象
	if (blobList.empty() && !rects.empty())
	{
		for (int i = 0; i < rects.size(); i++)
		{
            if (roi->intersects(rects[i]))
				addBlob(rects[i]);
		}
		return;
	}

	// 如果 blobList 不为空，rects 为空，则现有跟踪对象都要删除
	if (!blobList.empty() && rects.empty())
	{
		for (list<Ptr<Blob> >::iterator ptrBlob = blobList.begin(); ptrBlob != blobList.end(); ptrBlob++)
		{
            (*ptrBlob)->setToBeDeleted();
		}
		return;
	}

	// 其他情况
	// 检测是否有掉头
	if (configMatch.runCheckTurnAround)
	{
		for (list<Ptr<Blob> >::iterator ptrBlob = blobList.begin(); ptrBlob != blobList.end(); ptrBlob++)
		{
			Blob* pCurrBlob = *ptrBlob;
            if (pCurrBlob->getHistoryLength() % 5 == 0 &&
                pCurrBlob->doesTurnAround())
			{
#if CMPL_WRITE_CONSOLE
                printf("Blob ID: %d Turn Around\n", /*pCurrBlob->ID*/pCurrBlob->getID());
#endif
                pCurrBlob->setToBeDeleted();
			}
		}
	}

	int numOfRect = rects.size();
	int numOfBlob = blobList.size();

	// 当前帧中检测到的矩形的中心
	Point* centers = new Point[rects.size()];
	// 当前帧中的矩形和当前 blobList 中运动目标在上一帧中的矩形的交集和当前帧中矩形的面积的比值
	double* ratioToSelf = new double[blobList.size() * rects.size()];
	// 当前帧中的矩形和当前 blobList 中运动目标在上一帧中的矩形的交集和运动目标上一帧的矩形的面积的比值
	double* ratioToBlob = new double[blobList.size() * rects.size()];
	// 当前帧中的矩形中心和当前 blobList 中运动目标在上一帧中的矩形中心的距离
	double* dist = new double[blobList.size() * rects.size()];
	// 当前 blobList 中的运动目标和当前帧中的矩形的匹配关系
	bool* match = new bool[blobList.size() * rects.size()];
	// 是否对应新的运动目标的矩形
	bool* isNewBlobRect = new bool[rects.size()];

	// 计算当前帧中矩形的中心
	for (int i = 0; i < numOfRect; i++)
	{
		centers[i] = Point(rects[i].x + rects[i].width / 2, rects[i].y + rects[i].height / 2);
	}

	// 计算相交比例和距离
	int i = 0;
	for (list<Ptr<Blob> >::iterator ptrBlob = blobList.begin(); ptrBlob != blobList.end(); ptrBlob++, i++)
	{
		Blob* pCurrBlob = *ptrBlob;
        Rect lastRect = pCurrBlob->getCurrRect();
        Point lastCenter = Point(lastRect.x + lastRect.width / 2, lastRect.y + lastRect.height / 2);
		for (int j = 0; j < numOfRect; j++)
		{
			Rect intersectRect = lastRect & rects[j];
			ratioToSelf[i * numOfRect + j] = double(intersectRect.width * intersectRect.height) /
				double(rects[j].width * rects[j].height);
			ratioToBlob[i * numOfRect + j] = double(intersectRect.width * intersectRect.height) /
				double(lastRect.width * lastRect.height);
			dist[i * numOfRect + j] = sqrt(pow(double(lastCenter.x) - double(centers[j].x), 2) + 
				                      pow(double(lastCenter.y) - double(centers[j].y), 2));
		}
	}

	// 使用矩形距离进行匹配，一个矩形只能匹配一个运动目标
	for (int i = 0; i < numOfRect; i++)
	{
		// 如果当前矩形和观测线圈不相交
		if (!roi->intersects(rects[i]))
		{
			isNewBlobRect[i] = false;
			for (int j = 0; j < numOfBlob; j++)
			{
				match[j * numOfRect + i] = false;
			}
			continue;
		}

		// 下面处理当前矩形和观测线圈相交的情况
		// 先找和当前矩形距离最近的已跟踪目标
		float minDist = dist[i];
		int minDistIndex = 0; 
		for (int j = 1; j < numOfBlob; j++)
		{
			if (dist[j * numOfRect + i] < minDist)
			{
				minDist = dist[j * numOfRect + i];
				minDistIndex = j;
			}
		}
		// 没有能够和运动目标进行匹配的矩形，创建新的跟踪对象
		if (minDist > configMatch.maxDistRectAndBlob && 
			ratioToSelf[minDistIndex * numOfRect + i] < /*0.6*/configMatch.minRatioIntersectToSelf &&
			ratioToBlob[minDistIndex * numOfRect + i] < /*0.6*/configMatch.minRatioIntersectToBlob)
		{
			for (int j = 0; j < numOfBlob; j++)
			{
				match[j * numOfRect + i] = false;
			}
			isNewBlobRect[i] = true;
		}
		// 能够和运动目标匹配的矩形
		else
		{
			for (int j = 0; j < numOfBlob; j++)
			{
				if (j == minDistIndex)
					match[j * numOfRect + i] = true;
				else
					match[j * numOfRect + i] = false;
			}
			isNewBlobRect[i] = false;
		}
	}

#if CMPL_WRITE_CONSOLE
	if (configMatch.runDisplayCalcResults)
	{
		printf("rect     x     y     w     h\n");
		for (int i = 0; i < numOfRect; i++)
		{
			printf("%4d%6d%6d%6d%6d\n", i, rects[i].x, rects[i].y, rects[i].width, rects[i].height);
		}
		
		int i;
		printf("ratio to self matrix:\n");
		printf(" Blob ID");
		for (int i = 0; i < numOfRect; i++)
		{
			printf("   rect %2d", i);
		}
		printf("\n");
		
		i = 0;
		for (list<Ptr<Blob> >::iterator ptrBlob = blobList.begin(); ptrBlob != blobList.end(); ptrBlob++, i++)
		{
			Blob* pCurrBlob = *ptrBlob;
            printf("%8d", pCurrBlob->getID());
			for (int j = 0; j < numOfRect; j++)
			{
				printf("%10.2f", ratioToSelf[i * numOfRect + j]);
			}
			printf("\n");
		}

		printf("ratio to blob matrix:\n");
		printf(" Blob ID");
		for (int i = 0; i < numOfRect; i++)
		{
			printf("   rect %2d", i);
		}
		printf("\n");
		i = 0;
		for (list<Ptr<Blob> >::iterator ptrBlob = blobList.begin(); ptrBlob != blobList.end(); ptrBlob++, i++)
		{
			Blob* pCurrBlob = *ptrBlob;
			printf("%8d", pCurrBlob->getID());
			for (int j = 0; j < numOfRect; j++)
			{
				printf("%10.2f", ratioToBlob[i * numOfRect + j]);
			}
			printf("\n");
		}

		printf("dist matrix:\n");
		printf(" Blob ID");
		for (int i = 0; i < numOfRect; i++)
		{
			printf("   rect %2d", i);
		}
		printf("\n");
		i = 0;
		for (list<Ptr<Blob> >::iterator ptrBlob = blobList.begin(); ptrBlob != blobList.end(); ptrBlob++, i++)
		{
			Blob* pCurrBlob = *ptrBlob;
			printf("%8d", pCurrBlob->getID());
			for (int j = 0; j < numOfRect; j++)
			{
				printf("%10.2f", dist[i * numOfRect + j]);
			}
			printf("\n");
		}

		printf("match matrix:\n");
		printf(" Blob ID");
		for (int i = 0; i < numOfRect; i++)
		{
			printf("   rect %2d", i);
		}
		printf("\n");
		i = 0;
		for (list<Ptr<Blob> >::iterator ptrBlob = blobList.begin(); ptrBlob != blobList.end(); ptrBlob++, i++)
		{
			Blob* pCurrBlob = *ptrBlob;
			printf("%8d", pCurrBlob->getID());
			for (int j = 0; j < numOfRect; j++)
			{
				if (match[i * numOfRect + j])
					printf("      true");
				else
					printf("     false");
			}
			printf("\n");
		}
	}
#endif

	// 运动目标链表中的运动对象和矩形进行匹配
	i = 0;
	for (list<Ptr<Blob> >::iterator ptrBlob = blobList.begin(); ptrBlob != blobList.end(); ptrBlob++, i++)
	{
		Blob* pCurrBlob = *ptrBlob;

		// 已经被标记为删除的运动对象，将不进行匹配处理
		//if (pCurrBlob->isToBeDeleted == true)
        if (pCurrBlob->getIsToBeDeleted())
		{
			continue;
		}

		// 统计每个运动目标被几个矩形匹配
		int matchCount = 0;
		for (int j = 0; j < numOfRect; j++)
		{
			if (match[i * numOfRect + j])
				matchCount++;
		}

		int matchIndex = -1;
		// 运动目标没有任何矩形与之匹配
		if (matchCount == 0)
		{
            pCurrBlob->setToBeDeleted();
#if CMPL_WRITE_CONSOLE
            printf("Blob ID: %d Cannot match any rect in this frame.\n", pCurrBlob->getID());
#endif
			continue;
		}
		// 运动目标有超过一个矩形与之匹配
		else if (matchCount > 1)
		{
#if CMPL_WRITE_CONSOLE
            printf("Blob ID: %d Match multiple rects\n", pCurrBlob->getID());
#endif
			
			/*****************
			      直线拟合
			*****************/
			float minDist;
			int startIndex;

			int smallRectCount = 0;
			for (int j = 0; j < numOfRect; j++)
			{
				if (match[i * numOfRect + j])
					smallRectCount += (ratioToBlob[i * numOfRect + j] < 0.2 ? 1 : 0);
			}
			bool areAllRectsSmall = (smallRectCount == matchCount);
			
			Point pointInLine; 
			Point2d dirUnitVector;
			double avgError;
            vector<Point> centerHistory;
            pCurrBlob->getCenterHistory(centerHistory);
            linearRegres(centerHistory, pointInLine, dirUnitVector, avgError);
			// 如果历史轨迹小于 10 
			// 或者拟合出来的直线是竖直方向的直线
			// 或者轨迹上个点相距拟合直线的距离过大
			// 或者所有与运动目标相交的矩形面积都较小
			// 直接找最小距离
            if (pCurrBlob->getHistoryLength() < /*10*/configMatch.maxHistorySizeForDistMatch || 
				avgError > /*15*/configMatch.maxAvgErrorForDistMatch || areAllRectsSmall)
			{
				// 找第一个与运动目标匹配的矩形
				for (int j = 0; j < numOfRect; j++)
				{
					if (match[i * numOfRect + j])
					{
						minDist = dist[i * numOfRect + j];
						matchIndex = j;
						startIndex = j + 1;
						break;
					}
				}
				// 遍历剩下的与当前运动目标匹配的矩形，将和当前运动目标距离最近的矩形当做匹配矩形
				for (int j = startIndex; j < numOfRect; j++)
				{
					if (match[i * numOfRect + j])
					{
						if (dist[i * numOfRect + j] < minDist)
						{
							minDist = dist[i * numOfRect + j];
							matchIndex = j;
						}
					}
				}
				// 处理不能和当前运动目标匹配的当前帧矩形
				for (int j = 0; j < numOfRect; j++)
				{
					if (match[i * numOfRect + j] && j != matchIndex)
					{
						isNewBlobRect[j] = true;
					}
				}
			}
			// 其他情况，根据当前帧矩形中心和拟合直线的距离进行运动目标和矩形的匹配
			else
			{
				double* distToLine = new double[numOfRect];
				for (int j = 0; j < numOfRect; j++)
				{
					distToLine[j] = fabs(dirUnitVector.y * (centers[j].x - pointInLine.x) + dirUnitVector.x * (pointInLine.y - centers[j].y));
				}
#if CMPL_WRITE_CONSOLE
				if (configMatch.runDisplayCalcResults)
				{
					// 计算矩形的中心到直线的距离
					printf("dist to line:\n");
					printf("        ");					
					for (int j = 0; j < numOfRect; j++)
					{
						printf("   rect %2d", j);
					}
					printf("\n");
					printf("        ");	
					for (int j = 0; j < numOfRect; j++)
					{
						printf("%10.2f", distToLine[j]);
					}
					printf("\n");
				}
#endif
				
				// 找第一个与运动目标匹配的矩形
				for (int j = 0; j < numOfRect; j++)
				{
					if (match[i * numOfRect + j] && ratioToBlob[i * numOfRect + j] >= 0.2)
					{
						minDist = distToLine[j];
						matchIndex = j;
						startIndex = j + 1;
						break;
					}
				}
				// 遍历剩下的与当前运动目标匹配的矩形，将和当前运动目标距离最近的矩形当做匹配矩形
				for (int j = startIndex; j < numOfRect; j++)
				{
					if (match[i * numOfRect + j] && ratioToBlob[i * numOfRect + j] >= 0.2)
					{
						if (distToLine[j] < minDist)
						{
							minDist = distToLine[j];
							matchIndex = j;
						}
					}
				}
				delete [] distToLine;

				// 处理不能和当前运动目标匹配的当前帧矩形
				for (int j = 0; j < numOfRect; j++)
				{
					if (match[i * numOfRect + j] && j != matchIndex)
					{
						isNewBlobRect[j] = true;
					}
				}

#if CMPL_SHOW_IMAGE
				if (configMatch.runShowFitLine)
				{
					Mat lineImg = Mat::zeros(240, 320, CV_8UC1);
					line (lineImg, Point(pointInLine.x - 400 * dirUnitVector.x, pointInLine.y - 400 * dirUnitVector.y),
						           Point(pointInLine.x + 400 * dirUnitVector.x, pointInLine.y + 400 * dirUnitVector.y), Scalar(255));
					imshow("linear regression", lineImg);
					waitKey(0);
					destroyWindow("linear regression");
				}
#endif
			}			
		}
		// 运动目标只有一个矩形与之匹配
		else
		{
			for (int j = 0; j < numOfRect; j++)
			{
				if (match[i * numOfRect + j])
				{
					matchIndex = j;
					break;
				}
			}
		}

		// 判断当前矩形是否与观测区域有交集
		if (!roi->intersects(rects[matchIndex]))
		{
            pCurrBlob->setToBeDeleted();
		}
        else
        {
            pCurrBlob->setCurrRect(rects[matchIndex]);
        }
	}

	// 没有能够匹配的矩形，如果在观测区域以内，创建新的运动目标
	for (int i = 0; i < rects.size(); i++)
	{
		if (isNewBlobRect[i] && roi->intersects(rects[i]))
			addBlob(rects[i]);
	}
	
	delete [] centers;
	delete [] ratioToSelf;
	delete [] ratioToBlob;
	delete [] dist;
	delete [] match;
	delete [] isNewBlobRect;
}

void BlobTracker::BlobTrackerImpl::updateBlobListAfterCheck(void)
{
	for (list<Ptr<Blob> >::iterator ptrBlob = blobList.begin(); ptrBlob != blobList.end();)
	{
		Blob* pCurrBlob = *ptrBlob;
        if (pCurrBlob->getIsToBeDeleted())
		{
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
            printf("Blob ID: %d End tracking. Time stamp: %lld, Frame count: %d\n", pCurrBlob->getID(), *currTime, *currCount);
            pCurrBlob->printHistory();
#endif
			ptrBlob = blobList.erase(ptrBlob);
		}
		else
			ptrBlob++;
	}
}

void BlobTracker::BlobTrackerImpl::updateState(void)
{
	for (list<Ptr<Blob> >::iterator ptrBlob = blobList.begin(); ptrBlob != blobList.end(); ptrBlob++)
	{
	    (*ptrBlob)->updateState();
	}
}

void BlobTracker::BlobTrackerImpl::updateState(const Mat& origFrame, const Mat& foreImage)
{
	for (list<Ptr<Blob> >::iterator ptrBlob = blobList.begin(); ptrBlob != blobList.end(); ptrBlob++)
	{
	    (*ptrBlob)->updateState(origFrame, foreImage);
	}
}

void BlobTracker::BlobTrackerImpl::updateState(const Mat& origFrame, const Mat& foreImage, 
    const Mat& gradDiffImage, const Mat& lastGradDiffImage)
{
	for (list<Ptr<Blob> >::iterator ptrBlob = blobList.begin(); ptrBlob != blobList.end(); ptrBlob++)
	{
	    (*ptrBlob)->updateState(origFrame, foreImage, gradDiffImage, lastGradDiffImage);
	}
}

bool BlobTracker::BlobTrackerImpl::outputInfo(vector<ObjectInfo>& objects, bool isFinal) const
{
	bool isOutput = false;
	objects.clear();
    for (list<Ptr<Blob> >::const_iterator ptrBlob = blobList.begin(); ptrBlob != blobList.end(); ptrBlob++)
    {
        int index = objects.size();
        objects.push_back(ObjectInfo());
        if ((*ptrBlob)->outputInfo(objects[index], isFinal))
            isOutput = true;
    }
    return isOutput;
}

void BlobTracker::BlobTrackerImpl::drawTrackingState(Mat& frame, const Scalar& observedRegionColor, const Scalar& crossLoopOrLineColor,
    const Scalar& blobRectColor, const Scalar& blobHistoryColor) const
{
    roi->draw(frame, observedRegionColor);
    if (recordLoop) recordLoop->drawLoop(frame, crossLoopOrLineColor);
    if (recordLine) recordLine->drawLineSegment(frame, crossLoopOrLineColor);
    drawObjects(frame, blobRectColor);
    drawHistories(frame, blobHistoryColor);
}

void BlobTracker::BlobTrackerImpl::drawObjects(Mat& frame, const Scalar& color) const
{
	for (list<Ptr<Blob> >::const_iterator ptrBlob = blobList.begin(); ptrBlob != blobList.end(); ptrBlob++)
	{
		(*ptrBlob)->drawBlob(frame, color);
	}
}

void BlobTracker::BlobTrackerImpl::drawHistories(Mat& frame, const Scalar& color) const
{
	for (list<Ptr<Blob> >::const_iterator ptrBlob = blobList.begin(); ptrBlob != blobList.end(); ptrBlob++)
	{
	    (*ptrBlob)->drawHistory(frame, color);
	}
}

}