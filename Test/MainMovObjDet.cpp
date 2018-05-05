#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <iostream>
#include <fstream>
#include <exception>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "MovingObjectDetector.h"
#include "OperateGeometryTypes.h"

using namespace std;
using namespace cv;
using namespace zsfo;
using namespace ztool;

int main(int argc, char* argv[])
{
    string videoFileName = "D:\\Files\\RTXFiles\\zhengxuping\\bb.avi"
        /*"D:\\SHARED\\HaerbinVideo\\080011301522-20140919154529.asf"*/
        /*"D:/SHARED/XinjiangVideo/7S大湾北路金疆大厦卡口 20140730-112005--20140730-112205.mp4"*/;
    Size normSize(320, 240);
    int buildBackCount = 20;
    int updateBackInterval = 4;
    bool historyWithImages = false;
    int recordSnapshotMode = RecordSnapshotMode::CrossBottomBound;
    int saveSnapshotMode = SaveSnapshotMode::SaveSlice;
    int saveSnapshotInterval = 2, numOfSaved = 1;
    bool isNormScale = true;
    vector<vector<Point> > incPts(1);
    vector<vector<Point> > excPts;
    incPts[0].resize(4);
    incPts[0][0] = Point(10, 10);
    incPts[0][1] = Point(10, 230);
    incPts[0][2] = Point(310, 230);
    incPts[0][3] = Point(310, 10);
    vector<Point> catchPts(4);
    catchPts[0] = Point(10, 120);
    catchPts[1] = Point(10, 220);
    catchPts[2] = Point(310, 220);
    catchPts[3] = Point(310, 120);
    double minObjectArea = 50;
    double minObjectWidth = 10;
    double minObjectHeight = 10;
    bool charRegionCheck = false;
    vector<Rect> charRegions;
    bool checkTurnAround = true;
    double maxDistRectAndBlob = 30;
    double minRatioIntersectToSelf = 0.5;
    double minRatioIntersectToBlob = 0.5;

    VideoCapture cap;
    MovingObjectDetector mod;
    StampedImage input;
    ObjectDetails output;
    
    try
    {
        cap.open(videoFileName);
        input.time = cap.get(CV_CAP_PROP_POS_MSEC);
        input.number = cap.get(CV_CAP_PROP_POS_FRAMES);
        cap.read(input.image);
        mod.init(input, normSize, updateBackInterval, historyWithImages,
            recordSnapshotMode, saveSnapshotMode, saveSnapshotInterval, numOfSaved,
            isNormScale, incPts, excPts, catchPts, &minObjectArea, &minObjectWidth, &minObjectHeight,
            &charRegionCheck, charRegions, &checkTurnAround, &maxDistRectAndBlob,
            &minRatioIntersectToSelf, &minRatioIntersectToBlob);
        while (true)
        {
            input.time = cap.get(CV_CAP_PROP_POS_MSEC);
            input.number = cap.get(CV_CAP_PROP_POS_FRAMES);
            if (!cap.read(input.image))
                break;
            Mat normImage;
            resize(input.image, normImage, normSize);
            bool longWait = false;
            if (input.number < buildBackCount)
                mod.build(input);
            else
            {
                mod.proc(input, output);
                const vector<ObjectInfo>& objects = output.objects;
                printf("frame count %d:\n", input.number);
                int objSize = objects.size();
                for (int j = 0; j < objSize; j++)
                {
                    const ObjectInfo& refObj = objects[j];
                    printf("object count %3d, (%3d, %3d, %3d, %3d) %s\n",
                        j, refObj.currRect.x, refObj.currRect.y, refObj.currRect.width, refObj.currRect.height,
                        refObj.isFinal ? "end tracking" : "");
                    Rect normRect = mul(refObj.currRect, div(normSize, input.image.size()));
                    rectangle(normImage, normRect, Scalar(0, 255));
                    char idStr[32];
                    sprintf(idStr, "ID: %d", refObj.ID);
                    putText(normImage, idStr, normRect.tl(), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255));
                }
                printf("\n");                
                for (int j = 0; j < objSize; j++)
                {
                    const ObjectInfo& refObj = objects[j];                
                    if (refObj.isFinal && refObj.hasHistory)
                    {
                        /*printf("ID: %d\n", refObj.ID);
                        printf("      time     count    x    y    w    h\n");
                        int hisSize = refObj.history.size();
                        for (int k = 0; k < hisSize; k++)
                        {
                            const ObjectRecord& refRec = refObj.history[k];
                            printf("%10lld%10d%5d%5d%5d%5d\n", refRec.time, refRec.number, 
                                refRec.origRect.x, refRec.origRect.y, refRec.origRect.width, refRec.origRect.height);
                        }
                        printf("\n");*/
                    }
                    if (refObj.isFinal && refObj.hasSnapshotHistory)
                    {
                        char imageName[64];
                        printf("ID: %d, finished, output image frame count = %d\n", refObj.ID, refObj.snapshotHistory[0].number);
                        sprintf(imageName, "ID: %d", refObj.ID);
                        imshow(imageName, refObj.snapshotHistory[0].slice);
                        waitKey(0);
                        destroyWindow(imageName);
                        sprintf(imageName, "ID%d.bmp", refObj.ID);
                        imwrite(imageName, refObj.snapshotHistory[0].slice);
                        longWait = true;
                    }
                }
            }
            imshow("state", normImage);
            //waitKey(longWait ? 0 : 10);
            waitKey(10);
        }
        mod.final(output);
    }
    catch (exception& e)
    {
        printf("ERROR: %s\n", e.what());
        return 0;
    }   
    return 0;
}