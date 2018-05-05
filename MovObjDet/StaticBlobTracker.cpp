#include <fstream>

#include <opencv2/imgproc/imgproc.hpp>

#include "StaticBlobTracker.h"
#include "FileStreamScopeGuard.h"
#include "Exception.h"
#include "CompileControl.h"

using namespace std;
using namespace cv;
using namespace ztool;

namespace zsfo
{

void StaticBlob::init(const string& path)
{
    configCheckStatic = new ConfigCheckStatic;
    
    if (!path.empty())
    {
        fstream initFileStream;
        FileStreamScopeGuard<fstream> guard(initFileStream);
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
                THROW_EXCEPT("cannot find config params label [StaticBlob] for StaticBlob");
            }
        }
        while(string(stringNotUsed) != string("[StaticBlob]"));

        initFileStream >> stringNotUsed >> stringNotUsed;
        initFileStream >> configCheckStatic->minStaticTimeInMinute;

        initFileStream.close();
    }
    else
    {
        printf("StaticBlob is initialized with default param");
        configCheckStatic->minStaticTimeInMinute = 0.25;
    }

#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
    printf("display static blob config:\n");

    printf("  function check static:\n");
    printf("    configCheckStatic.minStaticTimeInMinute = %.4f\n", configCheckStatic->minStaticTimeInMinute);

    printf("\n");
#endif
}

void StaticBlob::init(int currID, const Rect& currRect, long long int time)
{
    ID = currID;
    rect = currRect;
    isStatic = false;
    hasOutputStatic = false;
    intervals.push_back(Interval(time, true));
}

void StaticBlob::setConfigParam(const double* minStaticTimeInMinute)
{
    if (minStaticTimeInMinute)
        configCheckStatic->minStaticTimeInMinute = *minStaticTimeInMinute;
}

void StaticBlob::checkStatic(const long long int time, const int count)
{
    if (isStatic) return;

    if (!intervals.back().isMatch) return;

    if (intervals.back().end - intervals.front().beg > configCheckStatic->minStaticTimeInMinute * 60 * 1000)
    {
        isStatic = true;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("Satic Blob ID: %d, static too long. Time stamp: %lld, Frame count: %d\n", ID, time, count);
#endif
        //waitKey(0);
    }
}

void StaticBlob::drawBlob(Mat& image, const Scalar& staticColor, const Scalar& nonStaticColor) const
{
    char name[10];
    sprintf(name, "SID:%d", ID);
    if (isStatic)
    {
        rectangle(image, rect, staticColor);
        putText(image, name, Point(rect.x + 1, rect.y + 20),
            CV_FONT_HERSHEY_SIMPLEX, 0.75, staticColor, 1);
    }
    else
    {
        rectangle(image, rect, nonStaticColor);
        putText(image, name, Point(rect.x + 1, rect.y + 20),
            CV_FONT_HERSHEY_SIMPLEX, 0.75, nonStaticColor, 1);
    }
}

void StaticBlob::displayHistory(void) const
{
    printf("Begin display history.............\n");
    printf("Number   match   begtime   endtime\n");
    for (int k = 0; k < intervals.size(); k++)
    {
        printf("%6d%s%10lld%10lld\n",
            k, intervals[k].isMatch ? "    true" : "   false", intervals[k].beg % 100000, intervals[k].end % 100000);
    }
    printf("End display history...............\n");
}

void StaticBlobTracker::init(const SizeInfo& sizesOrigAndNorm, const RegionOfInterest& observedRegion, const string& path)
{
    ptrImpl = new Impl;
    ptrImpl->init(sizesOrigAndNorm, observedRegion, path);
}

void StaticBlobTracker::setConfigParam(const double* allowedMissTimeInMinute, const double* minStaticTimeInMinute)
{
    ptrImpl->setConfigParam(allowedMissTimeInMinute, minStaticTimeInMinute);
}

void StaticBlobTracker::proc(const long long int time, const int count, 
    const vector<Rect>& rects, vector<StaticObjectInfo>& staticObjects)
{
    ptrImpl->proc(time, count, rects, staticObjects);
}

void StaticBlobTracker::drawBlobs(Mat& image, const Scalar& staticColor, const Scalar& nonStaticColor) const
{
    ptrImpl->drawBlobs(image, staticColor, nonStaticColor);
}

void StaticBlobTracker::Impl::init(const SizeInfo& sizesOrigAndNorm, const RegionOfInterest& observedRegion, const string& path)
{
    roi = observedRegion;
    sizeInfo = sizesOrigAndNorm;
    blobCount = 0;

    blobInstance.init(path);

    if (!path.empty())
    {
        fstream initFileStream;
        FileStreamScopeGuard<std::fstream> guard(initFileStream);
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
                THROW_EXCEPT("cannot find config params label [StaticBlobTracker] for StaticBlobTracker");
            }
        }
        while(string(stringNotUsed) != string("[StaticBlobTracker]"));

        initFileStream >> stringNotUsed >> stringNotUsed;
        initFileStream >> configUpdateBlobList.minMissTimeInMinuteToDelete;

        initFileStream.close();
    }
    else
    {
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("StaticBlobTracker is initialized with default param\n");
#endif
        configUpdateBlobList.minMissTimeInMinuteToDelete = 0.25;
    }

#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
    printf("display static blob config:\n");

    printf("  function check static:\n");
    printf("    configUpdateBlobList.minMissTimeInMinuteToDelete = %.4f\n", configUpdateBlobList.minMissTimeInMinuteToDelete);

    printf("\n");
#endif
}

void StaticBlobTracker::Impl::setConfigParam(const double* allowedMissTimeInMinute, const double* minStaticTimeInMinute)
{
    if (allowedMissTimeInMinute)
        configUpdateBlobList.minMissTimeInMinuteToDelete = *allowedMissTimeInMinute;
    blobInstance.setConfigParam(minStaticTimeInMinute);
}

void StaticBlobTracker::Impl::proc(const long long int time, const int count, 
    const vector<Rect>& rects, vector<StaticObjectInfo>& staticObjects)
{
    updateBlobList(time, count, rects);
    checkStatic(time, count);
    outputInfo(staticObjects);
}

void StaticBlobTracker::Impl::updateBlobList(const long long int time, const int count, const vector<Rect>& rects)
{
    if (rects.empty())
        return;

    int numOfValidRects = 0;
    vector<Rect> validRects;

    int numOfRects = rects.size();
    validRects.reserve(numOfRects);
    for (int i = 0; i < numOfRects; i++)
    {
        if (roi.intersects(rects[i]))
        {
            validRects.push_back(rects[i]);
            numOfValidRects++;
        }
    }

    //printf("  k  rect.x  rect.y  rect.w  rect.h\n");
    //for (int k = 0; k < numOfValidRects; k++)
    //{
    //  printf("%3d%8d%8d%8d%8d\n", k, validRects[k].x, validRects[k].y, validRects[k].width, validRects[k].height);
    //}
    //printf("SID  rect.x  rect.y  rect.w  rect.h\n");
    //for (list<StaticBlob*>::iterator itr = blobList.begin(); itr != blobList.end(); ++itr)
    //{
    //  printf("%3d%8d%8d%8d%8d\n", (*itr)->ID, 
    //      (*itr)->rect.x, (*itr)->rect.y, (*itr)->rect.width, (*itr)->rect.height);
    //}
    
    // 如果 blobList 为空，则将当前帧的 validRects 都 push 到 blobList 中，结束
    if (blobList.empty())
    {
        for (int i = 0; i < numOfValidRects; i++)
        {
            ++blobCount;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
            printf("Static Blob ID: %d Begin tracking. Time stamp: %lld, Frame count: %d\n", 
                blobCount, time, count);
#endif
            StaticBlob* staticBlob = new StaticBlob(blobInstance);
            staticBlob->init(blobCount, validRects[i], time);
            blobList.push_back(staticBlob);
        }
        return;
    }

    vector<RectInfo> rectInfos(numOfValidRects);
    for (int i = 0; i < numOfValidRects; i++)
    {
        rectInfos[i] = validRects[i];
    }

    // 如果 blobList 中的跟踪对象能够和 validRects 中的矩形完美匹配，则更新 blobList 中的那个对象
    for (list<StaticBlob*>::iterator itr = blobList.begin(); itr != blobList.end();)
    {
        bool match = false;
        for (int i = 0; i < numOfValidRects; i++)
        {
            if (rectInfos[i].isMatch)
                continue;

            Rect intersectRect = (*itr)->rect & rectInfos[i].rect;
            Rect unionRect = (*itr)->rect | rectInfos[i].rect;

            // 完美匹配的情况
            if (unionRect.width > 20 && unionRect.height > 20 ? 
                intersectRect.area() > 0.95 * unionRect.area() : intersectRect.area() > 0.75 * unionRect.area())
            {
                rectInfos[i].isMatch = true;
                match = true;
                (*itr)->rect = rectInfos[i].rect;
                if ((*itr)->intervals.back().isMatch)
                    (*itr)->intervals.back().end = time;
                else
                {
                    //printf("SID: %d appear again\n", (*itr)->ID);
                    //waitKey(0);
                    (*itr)->intervals.push_back(StaticBlob::Interval(time, true));
                }
                break;
            }
        }

        if (!match)
        {
            // 上一帧还处于被跟踪的状态 则新创建一个 Interval 表明当前没检测到
            if ((*itr)->intervals.back().isMatch)
            {
                //printf("SID: %d miss\n", (*itr)->ID);
                //waitKey(0);
                (*itr)->intervals.push_back(StaticBlob::Interval(time, false));
                ++itr;
            }
            // 上一帧处于丢失状态
            else
            {
                // 丢失时间太长，删除
                if (time - (*itr)->intervals.back().beg > 
                    configUpdateBlobList.minMissTimeInMinuteToDelete * 60 * 1000)
                {
                    //printf("SID: %d delete\n", (*itr)->ID);
                    //for (int k = 0; k < (*itr)->intervals.size(); k++)
                    //{
                    //  printf(" k = %d, match = %s, beg = %lld, end = %lld\n",
                    //      k, (*itr)->intervals[k].isMatch ? "true" : "false", (*itr)->intervals[k].beg, (*itr)->intervals[k].end);
                    //}
                    //printf("time: %lld, beg: %lld\n", time, (*itr)->intervals.back().beg);
                    //waitKey(0);
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
                    printf("Static Blob ID: %d Miss too long, deleted. Time stamp: %lld, Frame count: %d\n",
                        (*itr)->ID, time, count);
                    (*itr)->displayHistory();
                    //waitKey(0);
#endif
                    delete (*itr);
                    itr = blobList.erase(itr);                  
                }
                else
                {
                    (*itr)->intervals.back().end = time;
                    ++itr;
                }
            }
        }
        else
            ++itr;
    }

    // 处理没有被匹配上的矩形
    for (int i = 0; i < numOfValidRects; i++)
    {
        if (!rectInfos[i].isMatch)
        {
            ++blobCount;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
            printf("Static Blob ID: %d Begin tracking. Time stamp: %lld, Frame count: %d\n", 
                blobCount, time, count);
#endif
            StaticBlob* staticBlob = new StaticBlob(blobInstance);
            staticBlob->init(blobCount, validRects[i], time);
            blobList.push_back(staticBlob);
        }
    }
}

void StaticBlobTracker::Impl::checkStatic(const long long int time, const int count)
{
    // 判断是否停留时间过长
    for (list<StaticBlob*>::iterator itr = blobList.begin(); itr != blobList.end(); ++itr)
    {
        (*itr)->checkStatic(time, count);
    }
}

void StaticBlobTracker::Impl::outputInfo(vector<StaticObjectInfo>& staticObjects) const
{
    staticObjects.clear();
    for (list<StaticBlob*>::const_iterator itr = blobList.begin(); itr != blobList.end(); ++itr)
    {
        // 只有 isStatic 为真并且当前处于 match 状态的目标 并且尚未输出过静止状态 才会输出
        if ((*itr)->isStatic && (*itr)->hasOutputStatic == false &&(*itr)->intervals.back().isMatch)
        {
            (*itr)->hasOutputStatic = true;
            StaticObjectInfo objectInfo;
            objectInfo.ID = (*itr)->ID;
            objectInfo.rect = Rect((*itr)->rect.x * sizeInfo.horiScale,
                                   (*itr)->rect.y * sizeInfo.vertScale,
                                   (*itr)->rect.width * sizeInfo.horiScale,
                                   (*itr)->rect.height * sizeInfo.vertScale);
            staticObjects.push_back(objectInfo);
        }
    }
}

void StaticBlobTracker::Impl::drawBlobs(Mat& image, const Scalar& staticColor, const Scalar& nonStaticColor) const
{
    for (list<StaticBlob*>::const_iterator itr = blobList.begin(); itr != blobList.end(); ++itr)
    {
        (*itr)->drawBlob(image, staticColor, nonStaticColor);
    }
}

}