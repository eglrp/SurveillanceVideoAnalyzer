﻿#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "VisualInfo.h"
#include "BlobExtractor.h"
#include "RegionOfInterest.h"
#include "BlobTracker.h"

using namespace std;
using namespace cv;
using namespace zsfo;

int main(int argc, char** argv)
{
    const char* const path = /*"D:\\SHARED\\TrimpsVideo\\20140316\\01\\Video01_2014_03_16_14_57_28_type1.avi"*/
        /*"D:\\SHARED\\GuilinVideo\\DVR1_灵川八里街川东一路八里一路路口_20130826170046_20130826171424_63094171_0001.avi"*/
        /*"D:/SHARED/GuilinVideo/DVR1_灵川八里街川东一路八里一路路口_20130825150550_20130825151929_60634656_0001.avi"*/
        /*"D:/SHARED/GuilinVideo/DVR1_灵川八里街川东一路八里一路路口20130909070538.avi"*/
        /*"D:/SHARED/MiscellaneousVideo/16s_0.flv"*/
        "D:\\SHARED\\TaicangVideo\\2\\70.flv"
        /*"D:\\SHARED\\HongshenVideo\\顾戴2.avi"*/
        /*"D:\\SHARED\\HongshenVideo\\test1030.avi"*/;
    VideoCapture cap;
    cap.open(path);
    VisualInfo visualInfo;
    BlobExtractor blobExtractor;
    BlobTracker blobTracker;
    Size normSize(320, 240);    
    bool init = false;
    vector<Rect> stableRects;

    try
    {
    
    for (int i = 0; i < 200000000; i++)
    {
        Mat frame, image;
        long long int timeStamp = cap.get(CV_CAP_PROP_POS_MSEC);
        int frameCount = cap.get(CV_CAP_PROP_POS_FRAMES);
        if (!cap.read(frame)) break;
        resize(frame, image, normSize);
        printf("frame count = %d\n", i);

        if (!init)
        {
            init = true;

            visualInfo.init(image);

            blobExtractor.init(normSize);
            double minObjArea = 200, minObjWidth = 20, minObjHeight = 20;
            bool merge = true, mergeHori = true, mergeVert = true, mergeBigSmall = true;
            blobExtractor.setConfigParams(&minObjArea, &minObjWidth, &minObjHeight);
            /*blobExtractor.setConfigParams(&minObjArea, &minObjWidth, &minObjHeight,
                0, 0, vector<Rect>(), &merge, &mergeHori, &mergeVert, &mergeBigSmall);*/

            vector<vector<Point> > points(1);
            points[0].resize(4);
            points[0][0] = Point(10, 10);
            points[0][1] = Point(10, 230);
            points[0][2] = Point(310, 230);
            points[0][3] = Point(310, 10);

            RegionOfInterest roi;
            roi.init("roi", normSize, true, points);

            LineSegment lineSeg;
            lineSeg.init(Point(0, 0), Point(200, 200));

            VirtualLoop loop;
            loop.init("", points[0]);

            Size origSize = frame.size();
            SizeInfo sizeInfo;
            sizeInfo.create(origSize, normSize);

            //blobTracker.init(sizeInfo, roi, true);  // 只保存矩形历史记录, 无快照图片
            blobTracker.initLineSegment(sizeInfo, roi, lineSeg, SaveSnapshotMode::SaveScene | SaveSnapshotMode::SaveSlice, true);
            //blobTracker.initMultiRecord(sizeInfo, roi, SaveSnapshotMode::SaveScene | SaveSnapshotMode::SaveMask, 4, 4);
            //blobTracker.initTriBound(sizeInfo, roi, loop, SaveSnapshotMode::SaveScene | SaveSnapshotMode::SaveSlice);
            bool checkTurnAround = true;
            double maxDistRectAndBlob = 15;
            double intersectToSelf = 0.5;
            double intersectToBlob = 0.5;
            blobTracker.setConfigParams(&checkTurnAround, &maxDistRectAndBlob, &intersectToSelf, &intersectToBlob);

            continue;
        }

        Mat foreImage, backImage;
        visualInfo.update(image, foreImage, backImage, true, stableRects);
        imshow("image", image);
        imshow("orig fore image", foreImage);
        imshow("back image", backImage);
        
        vector<Rect> currRects;
        blobExtractor.proc(foreImage, Mat(), Mat(), currRects, stableRects);
        for (int j = 0; j < currRects.size(); j++)
            rectangle(image, currRects[j], Scalar(0, 0, 255), 2);
        for (int j = 0; j < stableRects.size(); j++)
            rectangle(image, stableRects[j], Scalar(0, 0, 0), 2);
        imshow("proc fore image", foreImage);

        vector<ObjectInfo> objects;
        //blobTracker.proc(timeStamp, frameCount, currRects, objects); // 只保存矩形历史记录, 无图片记录
        blobTracker.proc(frame, foreImage, timeStamp, frameCount, currRects, objects);
        Mat draw = image.clone();
        blobTracker.drawTrackingState(draw, Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255), Scalar(255, 255, 255));
        imshow("tracking state", draw);

        if (!objects.empty())
        {
            printf("frame count %d:\n", i);
            int objSize = objects.size();
            for (int j = 0; j < objSize; j++)
            {
                const ObjectInfo& refObj = objects[j];
                printf("object count %3d, (%3d, %3d, %3d, %3d) %s\n",
                    j, refObj.currRect.x, refObj.currRect.y, refObj.currRect.width, refObj.currRect.height,
                    refObj.isFinal ? "end tracking" : "");
            }
            printf("\n");
            bool wait = false;
            for (int j = 0; j < objSize; j++)
            {
                const ObjectInfo& refObj = objects[j];
                if (refObj.isFinal && refObj.hasSnapshotHistory)
                {
                    wait = true;
                    char buf[1024];
                    int shotHisSize = refObj.snapshotHistory.size();
                    for (int k = 0; k < shotHisSize; k++)
                    {
                        const ObjectSnapshotRecord& refRec = refObj.snapshotHistory[k];
                        if (refRec.scene.data)
                        {
                            Mat image(refRec.scene);
                            rectangle(image, refRec.rect, Scalar(255, 0, 0));
                            sprintf(buf, "Scene %d-%d", j, k);
                            imshow(buf, refRec.scene);
                        }
                        if (refRec.mask.data)
                        {
                            sprintf(buf, "Mask %d-%d", j, k);
                            imshow(buf, refRec.mask);
                        }
                        if (refRec.slice.data)
                        {
                            sprintf(buf, "Slice %d-%d", j, k);
                            imshow(buf, refRec.slice);
                        }
                    }
                }
                if (refObj.isFinal && refObj.hasHistory)
                {
                    wait = true;
                    printf("ID: %d\n", refObj.ID);
                    printf("      time     count    x    y    w    h\n");
                    int hisSize = refObj.history.size();
                    for (int k = 0; k < hisSize; k++)
                    {
                        const ObjectRecord& refRec = refObj.history[k];
                        printf("%10lld%10d%5d%5d%5d%5d\n", refRec.time, refRec.number, 
                            refRec.origRect.x, refRec.origRect.y, refRec.origRect.width, refRec.origRect.height);
                        if (refRec.image.data)
                        {
                            char buf[1024];
                            sprintf(buf, "history-ID-%d-count-%d", refObj.ID, k);
                            imshow(buf, refRec.image);
                        }
                    }
                    printf("\n");
                }
            }
            if (wait)
                waitKey(0);
            for (int j = 0; j < objSize; j++)
            {
                const ObjectInfo& refObj = objects[j];
                if (refObj.isFinal && refObj.hasSnapshotHistory)
                {
                    wait = true;
                    char buf[1024];
                    int shotHisSize = refObj.snapshotHistory.size();
                    for (int k = 0; k < shotHisSize; k++)
                    {
                        const ObjectSnapshotRecord& refRec = refObj.snapshotHistory[k];
                        if (refRec.scene.data)
                        {
                            sprintf(buf, "Scene %d-%d", j, k);
                            destroyWindow(buf);
                        }
                        if (refRec.mask.data)
                        {
                            sprintf(buf, "Mask %d-%d", j, k);
                            destroyWindow(buf);
                        }
                        sprintf(buf, "Slice %d-%d", j, k);
                        destroyWindow(buf);
                    }
                }
                if (refObj.isFinal && refObj.hasHistory)
                {
                    int hisSize = refObj.history.size();
                    for (int k = 0; k < hisSize; k++)
                    {
                        const ObjectRecord& refRec = refObj.history[k];
                        if (refRec.image.data)
                        {
                            char buf[1024];
                            sprintf(buf, "history-ID-%d-count-%d", refObj.ID, k);
                            destroyWindow(buf);
                        }
                    }
                }
            }
        }

        //waitKey(objects.empty() ? 1 : 0);
        //waitKey(i == 33260 ? 0 : 1);
        while (true)
        {
            char key = waitKey(25);
            if (key == ' ')
            {
                key = waitKey(0);
                if (key == ' ')
                    break;
            }
            else
                break;
        }

    }

    }
    catch (const std::exception& e)
    {
        printf("%s\n", e.what());
    }
    return 0;
}