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
#include "CreateDirectory.h"
#include "StringProcessor.h"
#include "CompileControl.h"

using namespace std;
using namespace cv;
using namespace zsfo;
using namespace ztool;

int main(int argc, char* argv[])
{   
    // 读取主函数的配置参数
    char videoFilePath[1000];
    char pathFilePath[1000];
    bool needCutVideo;
    bool isDebugMode;
    bool saveScene;
    bool saveSlice;
    bool saveMask;
    bool saveInfo;
    bool saveHistory;
    bool runShowImage;
    int waitTime;

    char stringNotUsed[1000];
    fstream argFile;
    argFile.open("MainArg.txt", ios::in);
    if (!argFile.is_open())
    {
        cout << "ERROR: cannot open file MainArg.txt" << "\n";
        exit(1);
    }
    argFile >> stringNotUsed >> videoFilePath;
    argFile >> stringNotUsed >> pathFilePath;
    argFile >> stringNotUsed >> needCutVideo;
    argFile >> stringNotUsed >> isDebugMode;
    argFile >> stringNotUsed >> saveScene;
    argFile >> stringNotUsed >> saveSlice;
    argFile >> stringNotUsed >> saveMask;
    argFile >> stringNotUsed >> saveInfo;
    argFile >> stringNotUsed >> saveHistory;
    argFile >> stringNotUsed >> runShowImage;
    argFile >> stringNotUsed >> waitTime;
    argFile.close();

    // 创建保存程序运行结果文件的路径
    createDirectory("result");
    // 提取视频文件名，不包括路径
    string videoFileName = videoFilePath;
    string validVideoName;
    cvtPathToFileName(videoFileName, validVideoName);
    
    int procEveryNFrame = 1;
    Size procSize(320, 240);
    int buildBackCount = 20;
    int updateBackInterval = 4;
    bool historyWithImages = false;
    int recordSnapshotMode = RecordSnapshotMode::Multi;
    int saveSnapshotMode = SaveSnapshotMode::SaveSlice;
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
        procVideo(videoFileName.c_str(), string("result/Result_Of_" + validVideoName).c_str(),
            saveScene ? "Scene" : "", saveSlice ? "Slice" : "", saveMask ? "Mask" : "",
            saveInfo ? "ObjectInfo.txt" : "", saveHistory ? "ObjectHistory.txt" : "", procEveryNFrame,
            procSize, buildBackCount, updateBackInterval, historyWithImages, 
            recordSnapshotMode, saveSnapshotMode, saveInterval, numOfSaved, 
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