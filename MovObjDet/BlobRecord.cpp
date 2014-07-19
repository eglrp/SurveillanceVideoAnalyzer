#include <cmath>
#include <algorithm>
#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>

#include "BlobTrackerInternal.h"
#include "OperateData.h"
#include "ShowData.h"
#include "Segment.h"
#include "FileStreamScopeGuard.h"
#include "Exception.h"
#include "CompileControl.h"

using namespace std;
using namespace cv;
using namespace ztool;

static const int BlobQuanHistoryCheckDirStep = 4;
static const int BlobQuanHistoryMaxDiffVal = 0;
static const int BlobVisHisRcrdFrmCntDiff = 10;
static const int CrsLnVisHisMaxDistToRecord = 15;
static const int CrsLnVisHisRcrdFrmCntDiff = 3;
const static int stepCheckStability = 5;

namespace zsfo
{

void BlobQuanRecord::makeRecord(const Rect& rect, double gradDiffMean, 
    long long int time, int count, const SizeInfo& sizeInfo)
{
    this->rect = rect;
    this->gradDiffMean = gradDiffMean;
    this->top.x = rect.x + rect.width / 2;
    this->top.y = rect.y;
    this->center.x = rect.x + rect.width / 2;
    this->center.y = rect.y + rect.height / 2;
    this->bottom.x = rect.x + rect.width / 2;
    this->bottom.y = rect.y + rect.height;
    this->origRect.x = this->rect.x * sizeInfo.horiScale;
    this->origRect.y = this->rect.y * sizeInfo.vertScale;
    this->origRect.width = this->rect.width * sizeInfo.horiScale;
    this->origRect.height = this->rect.height * sizeInfo.vertScale;
    this->count = count;
    this->time = time;
}

void BlobQuanRecord::makeRecord(const Mat& scene, const Rect& rect, 
    double gradDiffMean, long long int time, int count, const SizeInfo& sizeInfo)
{
    this->rect = rect;
    this->gradDiffMean = gradDiffMean;
    this->top.x = rect.x + rect.width / 2;
    this->top.y = rect.y;
    this->center.x = rect.x + rect.width / 2;
    this->center.y = rect.y + rect.height / 2;
    this->bottom.x = rect.x + rect.width / 2;
    this->bottom.y = rect.y + rect.height;
    this->origRect.x = this->rect.x * sizeInfo.horiScale;
    this->origRect.y = this->rect.y * sizeInfo.vertScale;
    this->origRect.width = this->rect.width * sizeInfo.horiScale;
    this->origRect.height = this->rect.height * sizeInfo.vertScale;
    this->count = count;
    this->time = time;
    scene(this->origRect).copyTo(this->image);
}

BlobQuanHistory::BlobQuanHistory(const cv::Ptr<SizeInfo>& sizesOrigAndNorm, 
    const cv::Ptr<long long int>& time, const cv::Ptr<int>& count, int blobID, bool historyWithImages)
    : ID(blobID),
      sizeInfo(sizesOrigAndNorm),
      currTime(time),
      currCount(count),
      checkDirStep(BlobQuanHistoryCheckDirStep),
      maxDiffVal(BlobQuanHistoryMaxDiffVal),
      recordImage(historyWithImages)
{

}

BlobQuanHistory::BlobQuanHistory(const BlobQuanHistory& history, int blobID)
    : ID(blobID),
      sizeInfo(history.sizeInfo),
      currTime(history.currTime),
      currCount(history.currCount),
      checkDirStep(history.checkDirStep),
      maxDiffVal(history.maxDiffVal),
      recordImage(history.recordImage)
{

}

BlobQuanHistory* BlobQuanHistory::createNew(int blobID) const
{
    BlobQuanHistory* ptr = new BlobQuanHistory(*this, blobID);
    return ptr;
}

int BlobQuanHistory::size(void) const
{
    return history.size();
}

void BlobQuanHistory::pushRecord(const Rect& rect, double gradDiffMean)
{
    currRecord.makeRecord(rect, gradDiffMean, *currTime, *currCount, *sizeInfo);
    history.push_back(currRecord);

    int index = history.size() - 1;
    if (index > 0 && index % checkDirStep == 0)
    {
        if (history[index].center.x + maxDiffVal < history[index - checkDirStep].center.x)
            dirCenterX.push_back(-1);
        else if (history[index].center.x > history[index- checkDirStep].center.x + maxDiffVal)
            dirCenterX.push_back(1);
        else
            dirCenterX.push_back(0);

        if (history[index].center.y + maxDiffVal < history[index - checkDirStep].center.y)
            dirCenterY.push_back(-1);
        else if (history[index].center.y > history[index - checkDirStep].center.y + maxDiffVal)
            dirCenterY.push_back(1);
        else
            dirCenterY.push_back(0);
    }
}

void BlobQuanHistory::pushRecord(const OrigSceneProxy& scene, const Rect& rect, double gradDiffMean)
{
    if (recordImage)
        currRecord.makeRecord(scene.getShallowCopy(), rect, gradDiffMean, *currTime, *currCount, *sizeInfo);
    else
        currRecord.makeRecord(rect, gradDiffMean, *currTime, *currCount, *sizeInfo);
    history.push_back(currRecord);

    int index = history.size() - 1;
    if (index > 0 && index % checkDirStep == 0)
    {
        if (history[index].center.x + maxDiffVal < history[index - checkDirStep].center.x)
            dirCenterX.push_back(-1);
        else if (history[index].center.x > history[index- checkDirStep].center.x + maxDiffVal)
            dirCenterX.push_back(1);
        else
            dirCenterX.push_back(0);

        if (history[index].center.y + maxDiffVal < history[index - checkDirStep].center.y)
            dirCenterY.push_back(-1);
        else if (history[index].center.y > history[index - checkDirStep].center.y + maxDiffVal)
            dirCenterY.push_back(1);
        else
            dirCenterY.push_back(0);
    }
}

void BlobQuanHistory::displayHistory(void) const
{
    if (history.empty()) return;
    printf("Begin display total history.........................................\n");
    printf("Number    Time   Count  Rect.x  Rect.y  Rect.w  Rect.h  GradDiffMean\n");
    for (int i = 0; i < history.size(); i++)
    {
        printf("%6d%8lld%8d", i + 1, history[i].time % 100000, history[i].count % 100000);
        printf("%8d%8d%8d%8d", history[i].rect.x, history[i].rect.y, history[i].rect.width, history[i].rect.height);
        printf("%14.4f\n", history[i].gradDiffMean);
    }
    printf("End display total history...........................................\n");
}

void BlobQuanHistory::outputHistory(ObjectInfo& objectInfo) const
{
    int size = history.size();
    objectInfo.hasHistory = size != 0;
    objectInfo.history.resize(history.size());
    for (int i = 0; i < history.size(); i++)
    {
        objectInfo.history[i].time = history[i].time;
        objectInfo.history[i].number = history[i].count;
        objectInfo.history[i].normRect = history[i].rect;
        objectInfo.history[i].origRect = history[i].origRect;
        objectInfo.history[i].image = history[i].image;
    }
}

void BlobQuanHistory::drawRect(Mat& normalImage, const Scalar& color) const
{
    //if (history.size() > 1)
        rectangle(normalImage, currRecord.rect, color);
    /*else
        rectangle(normalImage, currRecord.rect, Scalar(0, 255, 0));*/

    char name[10];
    sprintf(name, "ID:%d", ID);
    putText(normalImage, name, Point(currRecord.rect.x + 1, currRecord.rect.y - 5),
        CV_FONT_HERSHEY_SIMPLEX, 0.75, color, 1);
}

void BlobQuanHistory::drawTopHistory(Mat& normalImage, const Scalar& color) const
{
    if (history.size() > 1)
    {
        for (int i = 0; i < history.size() - 1; i++)
        {
            line(normalImage, history[i].top, history[i + 1].top, color);
        }
    }
}

void BlobQuanHistory::drawCenterHistory(Mat& normalImage, const Scalar& color) const
{
    if (history.size() > 1)
    {
        for (int i = 0; i < history.size() - 1; i++)
        {
            line(normalImage, history[i].center, history[i + 1].center, color);
        }
    }
}

void BlobQuanHistory::drawBottomHistory(Mat& normalImage, const Scalar& color) const
{
    if (history.size() > 1)
    {
        for (int i = 0; i < history.size() - 1; i++)
        {
            line(normalImage, history[i].bottom, history[i + 1].bottom, color);
        }
    }
}

void BlobQuanHistory::getCenterHistory(vector<Point>& centerHistory) const
{
    centerHistory.clear();
    if (history.empty()) return;
    int length = history.size();
    centerHistory.resize(length);
    for (int i = 0; i < length; i++)
        centerHistory[i] = history[i].center;
}

bool BlobQuanHistory::checkStability(int timeInMilliSec) const
{
    if (history.size() < 2)
        return false;

    int end = history.size() - 1, beg = -1;
    long long int begTime = history.back().time;
    // 根据历史的结束时间，往回退 timeInMilliSec 找到下标
    for (int i = history.size() - 1; i >= 0; i--)
    {
        if (history[i].time < begTime - timeInMilliSec)
        {
            beg = i;
            break;
        }
    }

    if (beg < 0) return false;

    if (end - beg <= stepCheckStability)
    {
        Rect begRect = history[beg].rect;
        Rect endRect = history[end].rect;
        Rect uniRect = begRect | endRect;
        double begRectRatio = double(begRect.area()) / double(uniRect.area());
        double endRectRatio = double(endRect.area()) / double(uniRect.area());

        if (((uniRect.width > 20 || uniRect.height > 20) ? 
             (begRectRatio > 0.9 && endRectRatio > 0.9) :
             (begRectRatio > 0.8 && endRectRatio > 0.8)) &&
            history[beg].gradDiffMean < 5 && history[end].gradDiffMean < 5)
            return true;
        else
            return false;
    }
    else
    {
        beg += stepCheckStability;
        for (int i = end; i >= beg; i -= stepCheckStability)
        {
            Rect begRect = history[i].rect;
            Rect endRect = history[i - stepCheckStability].rect;
            Rect uniRect = begRect | endRect;
            double begRectRatio = double(begRect.area()) / double(uniRect.area());
            double endRectRatio = double(endRect.area()) / double(uniRect.area());

            if (((uniRect.width > 20 || uniRect.height > 20) ? 
                 (begRectRatio > 0.9 && endRectRatio > 0.9) :
                 (begRectRatio > 0.8 && endRectRatio > 0.8)) &&
                history[i].gradDiffMean < 5 && history[i - stepCheckStability].gradDiffMean < 5)
                continue;
            else
                return false;
        }
        return true;
    }
}

//double BlobQuanHistory::calcImageSpeed(void) const
//{
//	int length = history.size();
//	if (length < 2)
//		return 0;
//
//	long long int begin = history[0].time;
//	long long int end = history[length - 1].time;
//	double timeElapsed = double(end - begin) / 1000;
//	if (timeElapsed == 0)
//		timeElapsed = 0.01;
//
//	double dist = 0;
//	const static int step = 5;
//	if (length <= step)
//	{
//		dist += sqrt(pow(double(history[0].bottom.x) - double(history[length - 1].bottom.x), 2) +
//				     pow(double(history[0].bottom.y) - double(history[length - 1].bottom.y), 2));	
//	}
//	else
//	{
//		int i;
//		for (i = 0; i < length - step; i += step)
//		{
//			dist += sqrt(pow(double(history[i].bottom.x) - double(history[i + step].bottom.x), 2) +
//						 pow(double(history[i].bottom.y) - double(history[i + step].bottom.y), 2));
//		}
//		if (i < length - 1)
//		{
//			dist += sqrt(pow(double(history[i].bottom.x) - double(history[length - 1].bottom.x), 2) +
//						 pow(double(history[i].bottom.y) - double(history[length - 1].bottom.y), 2));
//		}
//	}
//	return dist / timeElapsed;
//}
//
//double BlobQuanHistory::calcWorldSpeed(void) const
//{
//	int length = history.size();
//	if (length < 2)
//		return 0;
//
//	double accDist = 0;
//	const static int step = 5;
//	if (length <= step)
//	{
//		accDist += speedLoop->calcWorldDist(history[0].bottom, history[length - 1].bottom);
//	}
//	else
//	{
//		int i;
//		for (i = 0; i < length - step; i += step)
//		{
//			accDist += speedLoop->calcWorldDist(history[i].bottom, history[i + step].bottom);
//		}
//		if (i < length - 1)
//		{
//			accDist += speedLoop->calcWorldDist(history[i].bottom, history[length - 1].bottom);
//		}
//	}
//	accDist /= 100;
//	double timeElapsed = double(history[length - 1].time - history[0].time) / 1000;
//	if (timeElapsed == 0)
//		timeElapsed = 0.01;
//	return accDist / 100.0 / timeElapsed * 3.6;
//}

void BlobQuanHistory::linearRegres(Point& pointInLine, Point2d& dirVector, double& avgError) const
{
    double N = history.size();	
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    for (int i = 0; i < N; i++)
    {
        sumX += history[i].center.x;
        sumY += history[i].center.y;
        sumXY += history[i].center.x * history[i].center.y;
        sumX2 += history[i].center.x * history[i].center.x;
        sumY2 += history[i].center.y * history[i].center.y;
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
        sumError += fabs(dirVector.y * history[i].center.x + dirVector.x * pointInLine.y -
                         dirVector.y * pointInLine.x - dirVector.x * history[i].center.y);
    }
    avgError = sumError / N;
}

}

namespace
{
static bool checkDirTurnAround(vector<char>& dir)
{
    bool doesTurnAround = false;
    // 统计是正值多还是负值多
    int posiCount = 0, negaCount = 0;
    for (int i = 0 ; i < dir.size(); i++)
    {
        posiCount += dir[i] > 0 ? 1 : 0;
        negaCount += dir[i] < 0 ? 1 : 0;
    }
    // 如果正值多
    if (posiCount > negaCount)
    {
        // 统计前半段的正值
        int posiFstHalfCount = 0;
        for (int i = 0; i < dir.size() / 2; i++)
            posiFstHalfCount += dir[i] > 0 ? 1 : 0;
        // 如果前半段的正值多
        if (posiFstHalfCount > 0.7 * dir.size() / 2)
        {				
            // 统计最后一小段的负值
            int negaTailCount = 0;
            for (int i = dir.size() - 1; i >= dir.size() * 0.7; i--)
            {
                negaTailCount += dir[i] < 0 ? 1 : 0;
            }
            doesTurnAround = negaTailCount > 0.7 * 0.3 * dir.size();
        }
    }
    // 如果负值多
    else
    {
        // 统计前半段负值
        int negaFstHalfCount = 0;
        for (int i = 0; i < dir.size() / 2; i++)
            negaFstHalfCount += dir[i] < 0 ? 1 : 0;
        // 如果前半段的负值多
        if (negaFstHalfCount > 0.7 * dir.size() / 2)
        {				
            // 统计最后一小段的正值
            int posiTailCount = 0;
            for (int i = dir.size() - 1; i >= dir.size() * 0.7; i--)
            {
                posiTailCount += dir[i] > 0 ? 1 : 0;
            }
            doesTurnAround = posiTailCount > 0.7 * 0.3 * dir.size();
        }
    }
    return doesTurnAround;
}
}

namespace zsfo
{

bool BlobQuanHistory::checkTurnAround(void) const
{
    if (dirCenterX.size() < 10 || dirCenterY.size() < 10/* ||
        dirCenterX.size() % 5 != 0 || dirCenterY.size() % 5 != 0*/)
        return false;

    vector<char> dirX, dirY;
    ztool::localMedian(dirCenterX, dirX, 5);
    ztool::localMedian(dirCenterY, dirY, 5);
    //showArrayByHoriBar("x diff sign", dirX, false, true);
    //showArrayByVertBar("y diff sign", dirY, false, true);
    //waitKey(0);
    
    int zeroX = 0;
    for (int i = 0; i < dirX.size(); i++)
        zeroX += dirX[i] == 0 ? 1 : 0;
    double zeroRatioX = double(zeroX) / dirX.size();

    int zeroY = 0;
    for (int i = 0; i < dirY.size(); i++)
        zeroY += dirY[i] == 0 ? 1 : 0;
    double zeroRatioY = double(zeroY) / dirY.size();

    // 如果 0 值太多，则无法断定是否掉头，直接返回 false 
    if (zeroRatioX > 0.3 && zeroRatioY > 0.3)
        return false;

    return checkDirTurnAround(dirX) || checkDirTurnAround(dirY);
}

bool BlobQuanHistory::checkYDirection(int legalDirection) const
{
    if (dirCenterY.size() < 2)
        return true;

    int positiveCount = 0, negativeCount = 0, zeroCount = 0;
    for (int i = 0; i < dirCenterY.size(); i++)
    {
        if (dirCenterY[i] > 0)
            positiveCount++;
        else if (dirCenterY[i] < 0)
            negativeCount++;
        else
            zeroCount++;
    }
    if (positiveCount + negativeCount + zeroCount == 0)
        return true;

    double mean = double(positiveCount - negativeCount) / double(positiveCount + negativeCount);
    char currDirection;
    if (mean > 0.5)
        currDirection = 1;
    else if (mean < -0.5)
        currDirection = -1;
    else
        currDirection = 0;

    if (currDirection == 1 && legalDirection == 1)
        return true;
    else if (currDirection == 1 && legalDirection == 0)
        return false;
    else if (currDirection == -1 && legalDirection == 1)
        return false;
    else if (currDirection == -1 && legalDirection == 0)
        return true;
    else
        return true;
}

const Mat& OrigSceneProxy::getDeepCopy(void)
{
    if (!done) 
    {    
        done = true;
        shallowCopy.copyTo(deepCopy);
    }
    return deepCopy;
}

const Mat& OrigForeProxy::getDeepCopy(const Rect& normRect, const Rect& origRect)
{
    if (!done)
    {
        done = true;
        origFore = Mat::zeros(origSize, CV_8UC1);
    }
    Mat normForeROI = normFore(normRect);
    Mat origForeROI = origFore(origRect);
    resize(normForeROI, origForeROI, Size(origRect.width, origRect.height), 0, 0, INTER_NEAREST);
    return origFore;
}

}

static const int findDirection[5][3] = {{-1, -1, -1}, {-1, -1, -1}, {-1, 2, 1}, {-1, 1, 2}, {-1, 3, 4}};

namespace zsfo
{

void BlobSnapshotRecord::makeRecord(OrigSceneProxy& scene, OrigForeProxy& fore, 
    const SizeInfo& sizeInfo, const cv::Rect& baseRect, const cv::Rect& blobRect, 
    long long int currTime, int currCount, int saveMode)
{
    normRect = blobRect & baseRect;
    origRect = Rect(int(normRect.x * sizeInfo.horiScale), int(normRect.y * sizeInfo.vertScale),
                    int(normRect.width * sizeInfo.horiScale), int(normRect.height * sizeInfo.vertScale));
    time = currTime;
    count = currCount;

    // 保存全景图
    if (saveMode & SaveSnapshotMode::SaveScene)
        fullFrame = scene.getDeepCopy();

    // 保存运动目标的截图
    if (saveMode & SaveSnapshotMode::SaveSlice)
    {	    
        //Mat tempImage = Mat(scene.getDeepCopy(), origRect);
        //tempImage.copyTo(blobImage);
        blobImage = Mat(scene.getDeepCopy(), origRect);
    }

    // 保存运动目标的二值化前景图
    if (saveMode & SaveSnapshotMode::SaveMask)
    {
        //Mat foreImageROI = Mat(normForeImage, normRect);
        //resize(foreImageROI, foreImage, Size(origRect.width, origRect.height));
        //threshold(foreImage, foreImage, 127, 255, THRESH_BINARY);
        foreImage = fore.getDeepCopy(normRect, origRect);
    }
}

void BlobSnapshotRecord::makeRecord(OrigSceneProxy& scene, OrigForeProxy& fore, 
    const SizeInfo& sizeInfo, const Rect& baseRect, const Rect& blobRect, 
    int loopBound, int crossMode, long long int currTime, int currCount, int saveMode)
{
    bound = loopBound;
    crossIn = crossMode;
    direction = findDirection[bound + 1][crossIn + 1];
    makeRecord(scene, fore, sizeInfo, baseRect, blobRect, currTime, currCount, saveMode);
}

void BlobSnapshotRecord::copyTo(BlobSnapshotRecord& record) const
{
    record.bound = bound;
    record.crossIn = crossIn;
    record.direction = direction;
    record.normRect = normRect;
    record.origRect = origRect;
    record.time = time;
    record.count = count;
    blobImage.copyTo(record.blobImage);
    foreImage.copyTo(record.foreImage);
    fullFrame.copyTo(record.fullFrame);
}

void BlobSnapshotRecord::outputImages(ObjectSnapshotRecord& snapshotRecord) const
{
    // 保存时间戳
    snapshotRecord.time = time;
    // 保存帧编号
    snapshotRecord.number = count;
    // 保存矩形
    snapshotRecord.rect = origRect;
    // 保存截图时跨越虚拟线圈的位置
    snapshotRecord.bound = bound;
    // 保存截图时是进线圈还是出线圈
    snapshotRecord.cross = crossIn;
    // 保存截图的行驶方向
    snapshotRecord.direction = direction;
    // 保存抓拍车辆时的原始帧
    if (fullFrame.data)
        fullFrame.copyTo(snapshotRecord.scene);
    // 保存前景二值图
    if (foreImage.data)
        foreImage.copyTo(snapshotRecord.mask);
    // 保存原始帧中的车辆截图
    if (blobImage.data)
        blobImage.copyTo(snapshotRecord.slice);
}

BlobTriBoundSnapshotHistory::BlobTriBoundSnapshotHistory(const cv::Ptr<VirtualLoop>& catchLoop, const cv::Ptr<SizeInfo>& sizesOrigAndNorm, 
    const cv::Ptr<cv::Rect>& boundRect, const cv::Ptr<long long int>& time, const cv::Ptr<int>& count, 
    int blobID, int saveMode, const string& path)
    : ID(blobID),
      auxiCount(0),
      recordLoop(catchLoop),
      sizeInfo(sizesOrigAndNorm),
      baseRect(boundRect),
      currTime(time),
      currCount(count),
      hasUpdate(false),
      hasLeftCrossLoopLeft(false),
      hasRightCrossLoopRight(false),
      hasBottomCrossLoopBottom(false),
      configUpdate(new ConfigUpdate)
{
    if (!path.empty())
    {
        fstream initFileStream;
        ztool::FileStreamScopeGuard<fstream> guard(initFileStream);
        initFileStream.open(path.c_str());
        if (!initFileStream.is_open())
        {
            THROW_EXCEPT("cannot open file " + path);
        }
        char stringNotUsed[500];
        do
        {
            initFileStream >> stringNotUsed;
            if (initFileStream.eof())
            {
                THROW_EXCEPT("cannot find config params label [BlobVisualHistory] for BlobVisualHistory");
            }
        }
        while(string(stringNotUsed) != string("[BlobVisualHistory]"));
    
        initFileStream >> stringNotUsed >> stringNotUsed;
        initFileStream >> configUpdate->runShowImage;
        initFileStream >> stringNotUsed;
        initFileStream >> configUpdate->waitTime;

        initFileStream.close();
    }
    else
    {
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("BlobTriBoundVisualHistory is initialized with default param\n");
#endif
        configUpdate->runShowImage = true;
        configUpdate->waitTime = 0;
    }
    configUpdate->saveMode = saveMode;

#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
    printf("display blob tri bound visual history config:\n");

    printf("  function update:\n");
    printf("    configUpdate.runShowImage = %s\n", configUpdate->runShowImage ? "true" : "false");
    printf("    configUpdate.waitTime = %d\n", configUpdate->waitTime);
    printf("    configUpdate.saveMode = ");
    if (configUpdate->saveMode & SaveSnapshotMode::SaveScene)
        printf("scene ");
    if (configUpdate->saveMode & SaveSnapshotMode::SaveSlice)
        printf("slice ");
    if (configUpdate->saveMode & SaveSnapshotMode::SaveMask)
        printf("mask");
    printf("\n");

    printf("\n");
#endif
}

BlobTriBoundSnapshotHistory::BlobTriBoundSnapshotHistory(const BlobTriBoundSnapshotHistory& history, int blobID)
    : ID(blobID),
      auxiCount(0),
      recordLoop(history.recordLoop),
      sizeInfo(history.sizeInfo),
      baseRect(history.baseRect),
      currTime(history.currTime),
      currCount(history.currCount),
      configUpdate(history.configUpdate),
      hasUpdate(false),
      hasLeftCrossLoopLeft(false),
      hasRightCrossLoopRight(false),
      hasBottomCrossLoopBottom(false)
{

}

BlobTriBoundSnapshotHistory* BlobTriBoundSnapshotHistory::createNew(int blobID) const
{
    BlobTriBoundSnapshotHistory* ptr = new BlobTriBoundSnapshotHistory(*this, blobID);
    return ptr;
}

void BlobTriBoundSnapshotHistory::updateHistory(OrigSceneProxy& origFrame, 
    OrigForeProxy& foreImage, const cv::Rect& currRect)
{
    if (!hasUpdate)
    {
        auxiRecord.makeRecord(origFrame, foreImage, *sizeInfo, *baseRect,
                              currRect, -1, -1, *currTime, *currCount, configUpdate->saveMode);
        hasUpdate = true;
        lastRect = currRect;
        auxiCount = 0;
        return;
    }	
    else if (!hasLeftCrossLoopLeft &&
             !hasRightCrossLoopRight &&
             !hasBottomCrossLoopBottom)
    {
        auxiCount++;
        if (auxiCount == BlobVisHisRcrdFrmCntDiff)
        {
            auxiCount = 0;
            if (currRect.area() > lastRect.area())
            {			
                auxiRecord.makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, 
                                      currRect, -1, -1, *currTime, *currCount, configUpdate->saveMode);
            }
        }
    }

    if (hasLeftCrossLoopLeft == false)
    {
        Point currLeft = Point(currRect.x, currRect.y + currRect.height / 2);
        Point lastLeft = Point(lastRect.x, lastRect.y + lastRect.height / 2);
        if (recordLoop->leftToLeftBound(lastLeft) !=
            recordLoop->leftToLeftBound(currLeft))
        {
            hasLeftCrossLoopLeft = true;
            // 进入线圈
            if (recordLoop->leftToLeftBound(lastLeft))
            {
#if CMPL_WRITE_CONSOLE
                printf("Blob ID: %d Rect left crosses in record loop. "
                       "Time stamp: %lld. Frame count: %d\n", ID, *currTime, *currCount);
#endif
                leftRecord.makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, 
                                      currRect, 1, 1, *currTime, *currCount, configUpdate->saveMode);
            }
            // 离开线圈
            else
            {
#if CMPL_WRITE_CONSOLE
                printf("Blob ID: %d Rect left crosses out record loop. "
                       "Time stamp: %lld. Frame count: %d\n", ID, *currTime, *currCount);
#endif
                leftRecord.makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, 
                                      currRect, 1, 0, *currTime, *currCount, configUpdate->saveMode);
            }
#if CMPL_SHOW_IMAGE
            if (configUpdate->runShowImage)
            {
                imshow("left record", leftRecord.blobImage);
                waitKey(configUpdate->waitTime);
                destroyWindow("left record");
            }
#endif
        }
    }

    if (hasRightCrossLoopRight == false)
    {
        Point currRight = Point(currRect.x + currRect.width, currRect.y + currRect.height / 2);
        Point lastRight = Point(lastRect.x + lastRect.width, lastRect.y + lastRect.height / 2);
        if (recordLoop->rightToRightBound(lastRight) !=
            recordLoop->rightToRightBound(currRight))
        {
            hasRightCrossLoopRight = true;				
            // 进入线圈
            if (recordLoop->rightToRightBound(lastRight))
            {
#if CMPL_WRITE_CONSOLE
                printf("Blob ID: %d Rect right crosses in record loop. "
                       "Time stamp: %lld. Frame count: %d\n", ID, *currTime, *currCount);
#endif
                rightRecord.makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, 
                                        currRect, 2, 1, *currTime, *currCount, configUpdate->saveMode);
            }
            // 离开线圈
            else
            {
#if CMPL_WRITE_CONSOLE
                printf("Blob ID: %d Rect right crosses out record loop. "
                       "Time stamp: %lld. Frame count: %d\n", ID, *currTime, *currCount);
#endif
                rightRecord.makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, 
                                        currRect, 2, 0, *currTime, *currCount, configUpdate->saveMode);
            }
#if CMPL_SHOW_IMAGE
            if (configUpdate->runShowImage)
            {
                imshow("right record", rightRecord.blobImage);
                waitKey(configUpdate->waitTime);
                destroyWindow("right record");
            }
#endif
        }
    }

    if (hasBottomCrossLoopBottom == false)
    {
        Point currBottom = Point(currRect.x + currRect.width / 2, currRect.y + currRect.height);
        Point lastBottom = Point(lastRect.x + lastRect.width / 2, lastRect.y + lastRect.height);
        if (recordLoop->belowBottomBound(lastBottom) !=
            recordLoop->belowBottomBound(currBottom))
        {
            hasBottomCrossLoopBottom = true;				
            // 进入线圈
            if (recordLoop->belowBottomBound(lastBottom))
            {
#if CMPL_WRITE_CONSOLE
                printf("Blob ID: %d Rect bottom crosses in record loop. "
                       "Time stamp: %lld. Frame count: %d\n", ID, *currTime, *currCount);
#endif
                bottomRecord.makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, 
                                        currRect, 3, 1, *currTime, *currCount, configUpdate->saveMode);
            }
            // 离开线圈
            else
            {
#if CMPL_WRITE_CONSOLE
                printf("Blob ID: %d Rect bottom crosses out record loop. "
                       "Time stamp: %lld. Frame count: %d\n", ID, *currTime, *currCount);
#endif
                bottomRecord.makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, 
                                        currRect, 3, 0, *currTime, *currCount, configUpdate->saveMode);
            }
#if CMPL_SHOW_IMAGE
            if (configUpdate->runShowImage)
            {
                imshow("bottom record", bottomRecord.blobImage);
                waitKey(configUpdate->waitTime);
                destroyWindow("bottom record");
            }
#endif
        }
    }

    lastRect = currRect;
}

bool BlobTriBoundSnapshotHistory::outputHistory(ObjectInfo& objectInfo) const
{
    if (!hasUpdate)
    {
#if CMPL_WRITE_CONSOLE
        printf("Blob ID: %d visual history not updated, no image output.\n", ID);
#endif
        return false;
    }

    int count = 0;
    if (hasLeftCrossLoopLeft)
        count++;
    if (hasRightCrossLoopRight)
        count++;
    if (hasBottomCrossLoopBottom)
        count++;

#if CMPL_WRITE_CONSOLE
    printf("Blob ID: %d Output image.\n", ID);
#endif

    int label = 0;

    // 如果没抓拍到图片
    if (count == 0)
        label = 0;

    // 如果只抓拍到一张图片，则将这张图片输出
    if (count == 1)
    {
        if (hasLeftCrossLoopLeft) 
            label = 1;
        if (hasRightCrossLoopRight)
            label = 2;
        if (hasBottomCrossLoopBottom)
            label = 3;
    }

    // 如果抓拍到两张图片
    if (count == 2)
    {
        // 下边和左边跨越线圈
        if (hasBottomCrossLoopBottom &&
            hasLeftCrossLoopLeft)
        {
            // 先处理两个抓拍图片面积相差很大的情况
            /*if (bottomRecord.rect.area() > leftRecord.rect.area() * 4)
            {
                label = 3;
            }
            else if (leftRecord.rect.area() > bottomRecord.rect.area() * 4)
            {
                label = 1;
            }
            else*/
            {
                // 下边和左边都是出线圈
                if (!bottomRecord.crossIn && !leftRecord.crossIn)
                {
                    // 下边先出线圈
                    if (bottomRecord.time < leftRecord.time)
                        label = 3;
                    // 左边先出线圈
                    else
                        label = 1;
                }
                // 下边和左边都是进线圈
                else if (bottomRecord.crossIn && leftRecord.crossIn)
                {
                    // 下边先进线圈
                    if (bottomRecord.time < leftRecord.time)
                        label = 1;
                    // 左边先进线圈
                    else
                        label = 3;
                }
                // 下边进线圈，左边出线圈
                else if (bottomRecord.crossIn && !leftRecord.crossIn)
                {
                    label = 3;
                }
                // 下边出线圈，左边进线圈
                else
                {
                    label = 1;
                }
            }
        }
        // 下边和右边跨越线圈
        else if (hasBottomCrossLoopBottom &&
                 hasRightCrossLoopRight)
        {
            // 先处理两个抓拍图片面积相差很大的情况
            /*if (bottomRecord.rect.area() > rightRecord.rect.area() * 4)
            {
                label = 3;
            }
            else if (rightRecord.rect.area() > bottomRecord.rect.area() * 4)
            {
                label = 2;
            }
            else*/
            {
                // 下边和右边都是出线圈
                if (!bottomRecord.crossIn && !rightRecord.crossIn)
                {
                    // 下边先出线圈
                    if (bottomRecord.time < rightRecord.time)
                        label = 3;
                    // 右边先出线圈
                    else
                        label = 2;
                }
                // 下边和右边都是进线圈
                else if (bottomRecord.crossIn && rightRecord.crossIn)
                {
                    // 下边线进线圈
                    if (bottomRecord.time < rightRecord.time)
                        label = 2;
                    // 右边先进线圈
                    else
                        label = 3;
                }
                // 下边进线圈，左边出线圈
                else if (bottomRecord.crossIn && !rightRecord.crossIn)
                {
                    label = 3;
                }
                // 下边出线圈，左边进线圈
                else
                {
                    label = 2;
                }
            }
        }
        // 左边和右边跨越线圈
        else
        {
            if (leftRecord.normRect.area() > rightRecord.normRect.area())
                label = 1;
            else
                label = 2;
        }
    }

    // 如果抓拍到三张图片，则挑选最大的一张输出
    if (count == 3)
    {
        // 左进 右进 下进
        if (leftRecord.crossIn && rightRecord.crossIn && bottomRecord.crossIn)
        {
            // 哪张图片最后进线圈，就输出那张图片
            if (bottomRecord.time <= leftRecord.time && rightRecord.time <= leftRecord.time)
                label = 1;
            else if (bottomRecord.time <= rightRecord.time && leftRecord.time <= rightRecord.time)
                label = 2;
            else 
                label = 3;
        }
        // 左进 右进 下出
        else if (leftRecord.crossIn && rightRecord.crossIn && !bottomRecord.crossIn)
        {
            label = 3;
        }
        // 左进 右出 下进
        else if (leftRecord.crossIn && !rightRecord.crossIn && bottomRecord.crossIn)
        {
            if (bottomRecord.time <= leftRecord.time && leftRecord.time <= rightRecord.time)
                label = 1;
            else
                label = 3;
        }
        // 左进 右出 下出
        else if (leftRecord.crossIn && !rightRecord.crossIn && !bottomRecord.crossIn)
        {
            if (leftRecord.time <= rightRecord.time && rightRecord.time <= bottomRecord.time ||
                rightRecord.time <= bottomRecord.time && bottomRecord.time  <= leftRecord.time)
                label = 2;
            else 
                label = 3;
        }
        // 左出 右进 下进
        else if (!leftRecord.crossIn && rightRecord.crossIn && bottomRecord.crossIn)
        {
            if (bottomRecord.time <= rightRecord.time && rightRecord.time <= leftRecord.time)
                label = 2;
            else
                label = 3;
        }
        // 左出 右进 下出
        else if (!leftRecord.crossIn && rightRecord.crossIn && !bottomRecord.crossIn)
        {
            if (rightRecord.time <= leftRecord.time && leftRecord.time <= bottomRecord.time ||
                leftRecord.time <= bottomRecord.time && bottomRecord.time <= rightRecord.time)
                label = 1;
            else
                label = 3;
        }
        // 左出 右出 下进
        else if (!leftRecord.crossIn && !rightRecord.crossIn && bottomRecord.crossIn)
        {
            label = 3;
        }
        // 左出 右出 下出
        else
        {
            if (leftRecord.time <= rightRecord.time && leftRecord.time <= bottomRecord.time)
                label = 1;
            else if (rightRecord.time <= leftRecord.time && rightRecord.time <= bottomRecord.time)
                label = 2;
            else
                label = 3;
        }
    }

    objectInfo.hasSnapshotHistory = 1;
    objectInfo.snapshotHistory.resize(1);
    if (label == 0)
    {
#if CMPL_WRITE_CONSOLE
        printf("Blob ID: %d No cross loop record, auxiliary record selected for output.\n", ID);
#endif
        auxiRecord.outputImages(objectInfo.snapshotHistory[0]);
    }
    else if (label == 1)
    {
#if CMPL_WRITE_CONSOLE
        printf("Blob ID: %d Left record is optimal, selected for output.\n", ID);
#endif
        leftRecord.outputImages(objectInfo.snapshotHistory[0]);
    }
    else if (label == 2)
    {
#if CMPL_WRITE_CONSOLE
        printf("Blob ID: %d Right record is optimal, selected for output.\n", ID);
#endif
        rightRecord.outputImages(objectInfo.snapshotHistory[0]);
    }
    else
    {
#if CMPL_WRITE_CONSOLE
        printf("Blob ID: %d Bottom record is optimal, selected for output.\n", ID);
#endif
        bottomRecord.outputImages(objectInfo.snapshotHistory[0]);
    }
    return true;
}

BlobBottomBoundSnapshotHistory::BlobBottomBoundSnapshotHistory(const cv::Ptr<VirtualLoop>& catchLoop, const cv::Ptr<SizeInfo>& sizesOrigAndNorm, 
    const cv::Ptr<cv::Rect>& boundRect, const cv::Ptr<long long int>& time, const cv::Ptr<int>& count, 
    int blobID, int saveMode, const string& path)
    : ID(blobID),
      auxiCount(0),
      recordLoop(catchLoop),
      sizeInfo(sizesOrigAndNorm),
      baseRect(boundRect),
      currTime(time),
      currCount(count),
      hasUpdate(false),
      hasBottomCrossLoopBottom(false),
      configUpdate(new ConfigUpdate)
{
    if (!path.c_str())
    {
        fstream initFileStream;
        ztool::FileStreamScopeGuard<fstream> guard(initFileStream);
        initFileStream.open(path.c_str());
        if (!initFileStream.is_open())
        {
            THROW_EXCEPT("cannot open file " + path);
        }
        char stringNotUsed[500];
        do
        {
            initFileStream >> stringNotUsed;
            if (initFileStream.eof())
            {
                THROW_EXCEPT("cannot find config params label [BlobVisualHistory] for BlobVisualHistory");
            }
        }
        while(string(stringNotUsed) != string("[BlobVisualHistory]"));
    
        initFileStream >> stringNotUsed >> stringNotUsed;
        initFileStream >> configUpdate->runShowImage;
        initFileStream >> stringNotUsed;
        initFileStream >> configUpdate->waitTime;

        initFileStream.close();
    }
    else
    {
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("BlobBottomBoundVisualHistory is initialized with default param\n");
#endif
        configUpdate->runShowImage = true;
        configUpdate->waitTime = 0;
    }
    configUpdate->saveMode = saveMode;

#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
    printf("display blob bottom bound visual history config:\n");

    printf("  function update:\n");
    printf("    configUpdate.runShowImage = %s\n", configUpdate->runShowImage ? "true" : "false");
    printf("    configUpdate.waitTime = %d\n", configUpdate->waitTime);
    printf("    configUpdate.saveMode = ");
    if (configUpdate->saveMode & SaveSnapshotMode::SaveScene)
        printf("scene ");
    if (configUpdate->saveMode & SaveSnapshotMode::SaveSlice)
        printf("slice ");
    if (configUpdate->saveMode & SaveSnapshotMode::SaveMask)
        printf("mask");
    printf("\n");

    printf("\n");
#endif
}

BlobBottomBoundSnapshotHistory::BlobBottomBoundSnapshotHistory(const BlobBottomBoundSnapshotHistory& history, int blobID)
    : ID(blobID),
      auxiCount(0),
      recordLoop(history.recordLoop),
      sizeInfo(history.sizeInfo),
      baseRect(history.baseRect),
      currTime(history.currTime),
      currCount(history.currCount),
      configUpdate(history.configUpdate),
      hasUpdate(false),
      hasBottomCrossLoopBottom(false)
{

}

BlobBottomBoundSnapshotHistory* BlobBottomBoundSnapshotHistory::createNew(int blobID) const
{
    BlobBottomBoundSnapshotHistory* ptr = new BlobBottomBoundSnapshotHistory(*this, blobID);
    return ptr;
}

void BlobBottomBoundSnapshotHistory::updateHistory(OrigSceneProxy& origFrame, 
    OrigForeProxy& foreImage, const cv::Rect& currRect)
{
    if (!hasUpdate)
    {
        auxiRecord.makeRecord(origFrame, foreImage, *sizeInfo, *baseRect,
                              currRect, -1, -1, *currTime, *currCount, configUpdate->saveMode);
        hasUpdate = true;
        lastRect = currRect;
        auxiCount = 0;
        return;
    }
    else if (!hasBottomCrossLoopBottom)
    {
        auxiCount++;
        if (auxiCount == BlobVisHisRcrdFrmCntDiff)
        {
            auxiCount = 0;
            if (currRect.area() > lastRect.area())
            {
                auxiRecord.makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, 
                                      currRect, -1, -1, *currTime, *currCount, configUpdate->saveMode);
            }
        }
    }

    if (hasBottomCrossLoopBottom == false)
    {
        Point currBottom = Point(currRect.x + currRect.width / 2, currRect.y + currRect.height);
        Point lastBottom = Point(lastRect.x + lastRect.width / 2, lastRect.y + lastRect.height);
        if (recordLoop->belowBottomBound(lastBottom) !=
            recordLoop->belowBottomBound(currBottom))
        {
            hasBottomCrossLoopBottom = true;				
            // 进入线圈
            if (recordLoop->belowBottomBound(lastBottom))
            {
#if CMPL_WRITE_CONSOLE
                printf("Blob ID: %d Rect bottom crosses in record loop. "
                        "Time stamp: %lld. Frame count: %d\n", ID, *currTime, *currCount);
#endif
                bottomRecord.makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, 
                                        currRect, 3, 1, *currTime, *currCount, configUpdate->saveMode);
            }
            // 离开线圈
            else
            {
#if CMPL_WRITE_CONSOLE
                printf("Blob ID: %d Rect bottom crosses out record loop. "
                        "Time stamp: %lld. Frame count: %d\n", ID, *currTime, *currCount);
#endif
                bottomRecord.makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, 
                                        currRect, 3, 0, *currTime, *currCount, configUpdate->saveMode);
            }
#if CMPL_SHOW_IMAGE
            if (configUpdate->runShowImage)
            {
                imshow("bottom record", bottomRecord.blobImage);
                waitKey(configUpdate->waitTime);
                destroyWindow("bottom record");
            }
#endif
        }
    }

    lastRect = currRect;
}

bool BlobBottomBoundSnapshotHistory::outputHistory(ObjectInfo& objectInfo) const
{
    if (!hasUpdate)
    {
#if CMPL_WRITE_CONSOLE
        printf("Blob ID: %d visual history not updated, no image output.\n", ID);
#endif
        return false;
    }

    objectInfo.hasSnapshotHistory = 1;
    objectInfo.snapshotHistory.resize(1);
#if CMPL_WRITE_CONSOLE
    printf("Blob ID: %d Output image.\n", ID);
#endif
    if (hasBottomCrossLoopBottom)
    {
#if CMPL_WRITE_CONSOLE
        printf("Blob ID: %d Bottom record exists, output this record\n", ID);
#endif
        bottomRecord.outputImages(objectInfo.snapshotHistory[0]);
    }
    else
    {
#if CMPL_WRITE_CONSOLE
        printf("Blob ID: %d No cross loop record, auxiliary record selected for output.\n", ID);
#endif
        auxiRecord.outputImages(objectInfo.snapshotHistory[0]);
    }
    return true;
}

BlobCrossLineSnapshotHistory::BlobCrossLineSnapshotHistory(const cv::Ptr<LineSegment>& lineToCross, const cv::Ptr<SizeInfo>& sizesOrigAndNorm, 
    const cv::Ptr<cv::Rect>& boundRect, const cv::Ptr<long long int>& time, const cv::Ptr<int>& count, 
    int blobID, int saveMode, const string& path)
    : ID(blobID),
      auxiCount(0),
      recordLine(lineToCross),
      sizeInfo(sizesOrigAndNorm),
      baseRect(boundRect),
      currTime(time),
      currCount(count),
      hasUpdate(false),
      hasCrossLine(false),
      maxDistToRecord(CrsLnVisHisMaxDistToRecord),
      configUpdate(new ConfigUpdate)
{
    if (!path.empty())
    {
        fstream initFileStream;
        ztool::FileStreamScopeGuard<fstream> guard(initFileStream);
        initFileStream.open(path.c_str());
        if (!initFileStream.is_open())
        {
            THROW_EXCEPT("cannot open file " + path);
        }
        char stringNotUsed[500];
        do
        {
            initFileStream >> stringNotUsed;
            if (initFileStream.eof())
            {
                THROW_EXCEPT("cannot find config params label [BlobVisualHistory] for BlobVisualHistory");
            }
        }
        while(string(stringNotUsed) != string("[BlobVisualHistory]"));
    
        initFileStream >> stringNotUsed >> stringNotUsed;
        initFileStream >> configUpdate->runShowImage;
        initFileStream >> stringNotUsed;
        initFileStream >> configUpdate->waitTime;

        initFileStream.close();
    }
    else
    {
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("BlobCrossLineVisualHistory is initialized with default param\n");
#endif
        configUpdate->runShowImage = true;
        configUpdate->waitTime = 0;
    }
    configUpdate->saveMode = saveMode;

#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
    printf("display blob cross line visual history config:\n");

    printf("  function update:\n");
    printf("    configUpdate.runShowImage = %s\n", configUpdate->runShowImage ? "true" : "false");
    printf("    configUpdate.waitTime = %d\n", configUpdate->waitTime);
    printf("    configUpdate.saveMode = ");
    if (configUpdate->saveMode & SaveSnapshotMode::SaveScene)
        printf("scene ");
    if (configUpdate->saveMode & SaveSnapshotMode::SaveSlice)
        printf("slice ");
    if (configUpdate->saveMode & SaveSnapshotMode::SaveMask)
        printf("mask");
    printf("\n");

    printf("\n");
#endif
}

BlobCrossLineSnapshotHistory::BlobCrossLineSnapshotHistory(const BlobCrossLineSnapshotHistory& history, int blobID)
    : ID(blobID),
      auxiCount(0),
      recordLine(history.recordLine),
      sizeInfo(history.sizeInfo),
      baseRect(history.baseRect),
      currTime(history.currTime),
      currCount(history.currCount),
      configUpdate(history.configUpdate),
      hasUpdate(false),
      hasCrossLine(false),
      maxDistToRecord(history.maxDistToRecord)
{

}

BlobCrossLineSnapshotHistory* BlobCrossLineSnapshotHistory::createNew(int blobID) const
{
    BlobCrossLineSnapshotHistory* ptr = new BlobCrossLineSnapshotHistory(*this, blobID);
    return ptr;
}

void BlobCrossLineSnapshotHistory::updateHistory(OrigSceneProxy& origFrame, 
    OrigForeProxy& foreImage, const cv::Rect& currRect)
{
    int currDist = recordLine->distTo(Point(currRect.x + currRect.width / 2, currRect.y + currRect.height / 2));
    if (currDist < maxDistToRecord)
    {        
        if (!hasUpdate)
        {
#if CMPL_WRITE_CONSOLE
            printf("Blob ID: %d Rect close to line for the first time, dist is %d. "
                    "Time stamp: %lld. Frame count: %d\n", ID, currDist, *currTime, *currCount);
#endif
            crossLineRecord.makeRecord(origFrame, foreImage, *sizeInfo, *baseRect,
                currRect, -1, -1, *currTime, *currCount, configUpdate->saveMode);
            lastMinDist = currDist;
            hasUpdate = true; 
            hasCrossLine = true;
            auxiCount = 0;
#if CMPL_SHOW_IMAGE
            if (configUpdate->runShowImage)
            {
                imshow("cross line record", crossLineRecord.blobImage);
                waitKey(configUpdate->waitTime);
                destroyWindow("cross line record");
            }
#endif
            return;
        }
        auxiCount++;
        if (auxiCount == CrsLnVisHisRcrdFrmCntDiff)
        {
            auxiCount = 0;
            if (currDist < lastMinDist)
            {
#if CMPL_WRITE_CONSOLE
                printf("Blob ID: %d Rect closer to line than last record, curr dist is %d, last min dist is %d. "
                        "Time stamp: %lld. Frame count: %d\n", ID, currDist, lastMinDist, *currTime, *currCount);
#endif
                crossLineRecord.makeRecord(origFrame, foreImage, *sizeInfo, *baseRect,
                    currRect, -1, -1, *currTime, *currCount, configUpdate->saveMode);
                lastMinDist = currDist;
#if CMPL_SHOW_IMAGE
                if (configUpdate->runShowImage)
                {
                    imshow("cross line record", crossLineRecord.blobImage);
                    waitKey(configUpdate->waitTime);
                    destroyWindow("cross line record");
                }
#endif
            }   
        }
    }
}

bool BlobCrossLineSnapshotHistory::outputHistory(ObjectInfo& objectInfo) const
{
    if (!hasCrossLine)
    {
#if CMPL_WRITE_CONSOLE
        printf("Blob ID: %d Not cross line, no image output.\n", ID);
#endif
        return false;
    }
#if CMPL_WRITE_CONSOLE
    printf("Blob ID: %d Output image.\n", ID);
#endif
    objectInfo.hasSnapshotHistory = 1;
    objectInfo.snapshotHistory.resize(1);
    crossLineRecord.outputImages(objectInfo.snapshotHistory[0]);
    return true;
}

BlobMultiRecordSnapshotHistory::BlobMultiRecordSnapshotHistory(const cv::Ptr<SizeInfo>& sizesOrigAndNorm, 
    const cv::Ptr<cv::Rect>& boundRect, const cv::Ptr<long long int>& time, const cv::Ptr<int>& count, 
    int blobID, int saveMode, int saveInterval, int numOfSaved, const string& path)
    : ID(blobID),
      auxiCount(0),
      allInside(false),
      sizeInfo(sizesOrigAndNorm),
      baseRect(boundRect),
      currTime(time),
      currCount(count),
      configUpdate(new ConfigUpdate)
{
    if (!path.empty())
    {
        fstream initFileStream;
        ztool::FileStreamScopeGuard<fstream> guard(initFileStream);
        initFileStream.open(path.c_str());
        if (!initFileStream.is_open())
        {
            THROW_EXCEPT("cannot open file " + path);
        }
        char stringNotUsed[500];
        do
        {
            initFileStream >> stringNotUsed;
            if (initFileStream.eof())
            {
                THROW_EXCEPT("cannot find config params label [BlobVisualHistory] for BlobVisualHistory");
            }
        }
        while(string(stringNotUsed) != string("[BlobVisualHistory]"));
    
        initFileStream >> stringNotUsed >> stringNotUsed;
        initFileStream >> configUpdate->runShowImage;
        initFileStream >> stringNotUsed;
        initFileStream >> configUpdate->waitTime;

        initFileStream.close();
    }
    else
    {
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("BlobMultiRecordVisualHistory is initialized with default param\n");
#endif
        configUpdate->runShowImage = true;
        configUpdate->waitTime = 0;
    }
    configUpdate->saveMode = saveMode;
    configUpdate->saveInterval = saveInterval;
    configUpdate->numOfSaved = numOfSaved;
    history.reserve(configUpdate->numOfSaved);

#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
    printf("display blob multi record visual history config:\n");

    printf("  function update:\n");
    printf("    configUpdate.runShowImage = %s\n", configUpdate->runShowImage ? "true" : "false");
    printf("    configUpdate.waitTime = %d\n", configUpdate->waitTime);
    printf("    configUpdate.saveMode = ");
    if (configUpdate->saveMode & SaveSnapshotMode::SaveScene)
        printf("scene ");
    if (configUpdate->saveMode & SaveSnapshotMode::SaveSlice)
        printf("slice ");
    if (configUpdate->saveMode & SaveSnapshotMode::SaveMask)
        printf("mask");
    printf("\n");
    printf("    configUpdate.saveInterval = %d\n", configUpdate->saveInterval);
    printf("    configUpdate.numOfSaved = %d\n", configUpdate->numOfSaved);

    printf("\n");
#endif
}

BlobMultiRecordSnapshotHistory::BlobMultiRecordSnapshotHistory(const BlobMultiRecordSnapshotHistory& history, int blobID)
    : ID(blobID),
      auxiCount(0),
      allInside(false),
      sizeInfo(history.sizeInfo),
      baseRect(history.baseRect),
      currTime(history.currTime),
      currCount(history.currCount),
      configUpdate(history.configUpdate)
{
    this->history.reserve(configUpdate->numOfSaved);
}

BlobMultiRecordSnapshotHistory* BlobMultiRecordSnapshotHistory::createNew(int blobID) const
{
    BlobMultiRecordSnapshotHistory* ptr = new BlobMultiRecordSnapshotHistory(*this, blobID);
    return ptr;
}

} // namespace zsfo

namespace
{

struct AbsolutelyInside
{
    AbsolutelyInside(const Rect& rect)
        : region(Rect(rect.x + 5, rect.y + 5, rect.width - 10, rect.height - 10))
    {}
    bool operator()(const Rect& rect)
    {
        return region.contains(rect.tl()) && region.contains(rect.br());
    }
    bool operator()(const zsfo::BlobSnapshotRecord& record)
    {
        return region.contains(record.normRect.tl()) && region.contains(record.normRect.br());
    }
    Rect region;
};

struct NotAbsolutelyInside
{
    NotAbsolutelyInside(const Rect& rect)
        : region(Rect(rect.x + 5, rect.y + 5, rect.width - 10, rect.height - 10))
    {}
    bool operator()(const Rect& rect)
    {
        return !region.contains(rect.tl()) || !region.contains(rect.br());
    }
    bool operator()(const zsfo::BlobSnapshotRecord& record)
    {
        return !region.contains(record.normRect.tl()) || !region.contains(record.normRect.br());
    }
    Rect region;
};

struct Less
{
    bool operator()(const zsfo::BlobSnapshotRecord& lhs, const zsfo::BlobSnapshotRecord& rhs)
    {
        return lhs.normRect.area() < rhs.normRect.area();
    }
};

}

namespace zsfo
{

void BlobMultiRecordSnapshotHistory::updateHistory(OrigSceneProxy& origFrame,
    OrigForeProxy& foreImage, const cv::Rect& currRect)
{
    if (history.empty())
    {
        history.push_back(BlobSnapshotRecord());
        history[0].makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, currRect,
            *currTime, *currCount, configUpdate->saveMode);
        if (configUpdate->numOfSaved == 1)
        {
            AbsolutelyInside inside(Rect(0, 0, sizeInfo->normWidth, sizeInfo->normHeight));
            allInside = inside(history[0]);
        }
        auxiCount = 0;
        return;
    }
    auxiCount++;
    if (auxiCount == configUpdate->saveInterval)
    {
        auxiCount = 0;
        int size = history.size();
        AbsolutelyInside inside(Rect(0, 0, sizeInfo->normWidth, sizeInfo->normHeight));
        if (size < configUpdate->numOfSaved)
        {
            history.push_back(BlobSnapshotRecord());
            history[size].makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, currRect,
                *currTime, *currCount, configUpdate->saveMode);
            if (size + 1 == configUpdate->numOfSaved &&
                count_if(history.begin(), history.end(), inside) == configUpdate->numOfSaved)
                allInside = true;
        }
        else
        {
            // 如果只使用下面这段比较短的代码, 则保存的是面积最大的那些截图,
            // 由于近大远小的关系, 这其中有些截图很可能贴着画面的边界, 并且只截到运动物体的一部分
            //vector<BlobVisualRecord>::iterator itrMin = min_element(history.begin(), history.end(), Less());
            //if (currRect.area() > itrMin->normRect.area())
            //{
            //    itrMin->makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, currRect,
            //        *currTime, *currCount, configUpdate->saveMode);
            //}
            
            // 所有的截图都不靠近画面边界
            if (allInside)
            {
                // 如果当前矩形不靠近画面边界
                if (inside(currRect))
                {      
                    vector<BlobSnapshotRecord>::iterator itrMin = min_element(history.begin(), history.end(), Less());
                    // 并且面积大于面积最小的历史截图, 则用当前帧的截图替换
                    if (currRect.area() > itrMin->normRect.area())
                    {
                        itrMin->makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, currRect,
                            *currTime, *currCount, configUpdate->saveMode);
                    }
                }
                // 如果当前矩形靠近画面边界, 则不再考虑保存截图
            }
            // 至少有一张截图靠近画面边界
            else
            {
                int numOfInside = count_if(history.begin(), history.end(), inside);
                // 如果当前矩形不靠近画面边界, 无论如何都要将当前帧的截图塞到历史截图向量中
                if (inside(currRect))
                {
                    // 所有截图都靠近画面边界, 选出历史截图中面积最小的进行替换
                    if (numOfInside == 0)
                    {                        
                        vector<BlobSnapshotRecord>::iterator itrMin = min_element(history.begin(), history.end(), Less());
                        itrMin->makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, currRect,
                            *currTime, *currCount, configUpdate->saveMode);
                    }
                    // 不止一张截图靠近画面边界, 选出靠近画面边界的历史截图中面积最小的进行替换
                    else if (numOfInside < configUpdate->numOfSaved - 1)
                    {
                        sort(history.begin(), history.end(), Less());
                        NotAbsolutelyInside notInside(Rect(0, 0, sizeInfo->normWidth, sizeInfo->normHeight));
                        vector<BlobSnapshotRecord>::iterator itrMin = find_if(history.begin(), history.end(), notInside);
                        itrMin->makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, currRect,
                            *currTime, *currCount, configUpdate->saveMode);
                    }
                    // 恰有一张截图靠近画面边界, 替换之
                    else if (numOfInside == configUpdate->numOfSaved - 1)
                    {
                        NotAbsolutelyInside notInside(Rect(0, 0, sizeInfo->normWidth, sizeInfo->normHeight));
                        vector<BlobSnapshotRecord>::iterator itr = find_if(history.begin(), history.end(), notInside);
                        itr->makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, currRect,
                            *currTime, *currCount, configUpdate->saveMode);
                    }
                    allInside = (numOfInside + 1 == configUpdate->numOfSaved);
                }
                // 当前矩形靠近画面边界
                else
                {
                    // 所有截图都靠近画面边界, 选出历史截图中面积最小的进行替换
                    if (numOfInside == 0)
                    {
                        vector<BlobSnapshotRecord>::iterator itrMin = min_element(history.begin(), history.end(), Less());
                        if (currRect.area() > itrMin->normRect.area())
                        {
                            itrMin->makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, currRect,
                                *currTime, *currCount, configUpdate->saveMode);
                        }
                    }
                    // 不是所有的截图都靠近画面边界, 选出靠近画面边界的历史截图中面积最小的进行替换
                    else
                    {
                        sort(history.begin(), history.end(), Less());
                        NotAbsolutelyInside notInside(Rect(0, 0, sizeInfo->normWidth, sizeInfo->normHeight));
                        vector<BlobSnapshotRecord>::iterator itrMin = find_if(history.begin(), history.end(), notInside);
                        if (currRect.area() > itrMin->normRect.area())
                        {
                            itrMin->makeRecord(origFrame, foreImage, *sizeInfo, *baseRect, currRect,
                                *currTime, *currCount, configUpdate->saveMode);
                        }
                    }
                }
            }
        }
    }
}

bool BlobMultiRecordSnapshotHistory::outputHistory(ObjectInfo& objectInfo) const
{
    int size = history.size();
    objectInfo.hasSnapshotHistory = (size != 0);
    objectInfo.snapshotHistory.resize(size);
    for (int i = 0; i < size; i++)
        history[i].outputImages(objectInfo.snapshotHistory[i]);
    return size != 0;
}

}
