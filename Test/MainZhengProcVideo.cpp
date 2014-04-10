#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <exception>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "MovingObjectDetector.h"

using namespace std;
using namespace cv;
using namespace zsfo;

int	main(int argc, char* argv[])
{
	string videoFileName = "D:/SHARED/TaicangVideo/1/70.flv";
	string savePath = "result";
	bool saveScene = true;
	bool saveSlice = true;
	bool saveMask = false;
	bool saveInfo = false;
	bool saveHistory = true;
    int procEveryNFrame = 1;
    Size procSize(320, 240);
    int buildBackCount = 20;
    int updateBackInterval = 4;
    int recordMode = RecordMode::MultiVisualRecord;
    int saveMode = SaveImageMode::SaveSlice;
    int saveInterval = 2, numOfSaved = 4;
    bool isNormScale = true;
    vector<vector<Point> > incPts(1);
    vector<vector<Point> > excPts;
    /*incPts[0].resize(2);
    incPts[0][0] = Point(10, 120);
    incPts[0][1] = Point(310, 230);*/
    incPts[0].resize(4);
    /*incPts[0][0] = Point(10, 300);
    incPts[0][1] = Point(10, 570);
    incPts[0][2] = Point(700, 570);
    incPts[0][3] = Point(700, 250);*/
    incPts[0][0] = Point(10, 120);
    incPts[0][1] = Point(10, 230);
    incPts[0][2] = Point(310, 230);
    incPts[0][3] = Point(310, 120);
    vector<Point> catchPts;
    /*catchPts = incPts[0];*/
    /*catchPts[0] = Point(0, 0);
    catchPts[1] = Point(500, 500);*/
    double minObjectArea = 50;
    double minObjectWidth = 10;
    double minObjectHeight = 10;
    bool charRegionCheck = false;
    vector<Rect> charRegions;
    bool checkTurnAround = true;
    double maxDistRectAndBlob = 20;
    double minRatioIntersectToSelf = 0.5;
    double minRatioIntersectToBlob = 0.5;
    try
    {
        procVideo(videoFileName.c_str(), savePath.c_str(),
            saveScene ? "Scene" : "", saveSlice ? "Slice" : "", saveMask ? "Mask" : "",
            saveInfo ? "ObjectInfo.txt" : "", saveHistory ? "ObjectHistory.txt" : "", procEveryNFrame,
            procSize, buildBackCount, updateBackInterval, recordMode, saveMode, saveInterval, numOfSaved, 
            isNormScale, incPts, excPts, catchPts, &minObjectArea, &minObjectWidth, &minObjectHeight,
            &charRegionCheck, charRegions, &checkTurnAround, &maxDistRectAndBlob, 
            &minRatioIntersectToSelf, &minRatioIntersectToBlob);
    }
    catch (const exception& e)
    {
        cout << e.what() << "\n";
    }
    system("pause");
    return 0;
}