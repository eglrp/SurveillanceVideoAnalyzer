﻿#include <climits>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ProcVideo.h"
#include "MovingObjectDetector.h"
#include "OperateGeometryTypes.h"
#include "CreateDirectory.h"
#include "Exception.h"
#include "CompileControl.h"

using namespace std;
using namespace cv;

namespace
{
template <typename T>
string getString(T val)
{
    stringstream strm;
    strm << val;
    return strm.str();
}
}

namespace oip
{
struct ObjectInfoParser
{
    void init(const string& saveImageDir, const string& listFileName,
        const string& scenePrefix, const string& slicePrefix,
        const string& saveHistoryDir, const string& historyFileName);
    void parse(const vector<zsfo::ObjectInfo>& src, vector<zpv::ObjectInfo>& dst);
    void final(void);

    int objectCount;
    std::string imageDir;
    //std::string listName;
    std::string sceneNamePrefix;
    std::string sliceNamePrefix;
    std::string historyName;
};

void ObjectInfoParser::init(const string& saveImageDir, const string& listFileName,
    const string& scenePrefix, const string& slicePrefix,
    const string& saveHistoryDir, const string& historyFileName)
{
    objectCount = 0;
    imageDir = saveImageDir + "/";
    //listName = saveImageDir + "/" + listFileName;
    sceneNamePrefix = saveImageDir + "/" + scenePrefix;
    sliceNamePrefix = saveImageDir + "/" + slicePrefix;
    historyName = saveHistoryDir + "/" + historyFileName;

    ztool::createDirectory(saveImageDir);
    ztool::createDirectory(saveHistoryDir);

    fstream file;
    //file.open(listName.c_str(), ios::out);
    //file << "      ID        Time       Count       X       Y       W       H\n";
    //file.close();
    file.open(historyName.c_str(), ios::out);
    file.close();
}

}

static void getNormHistoryProperties(const vector<zsfo::ObjectRecord>& history, 
    double& avgWidth, double& avgHeight, double& stdDevX, double& stdDevY)
{
    avgWidth = 0;
    avgHeight = 0;
    stdDevX = 0;
    stdDevY = 0;
    if (history.empty()) return;

    int length = history.size();
    Mat xMat(1, length, CV_64FC1), yMat(1, length, CV_64FC1);
    Mat widthMat(1, length, CV_64FC1), heightMat(1, length, CV_64FC1);    
    double* ptrX = (double*)xMat.data;
    double* ptrY = (double*)yMat.data;
    double* ptrWidth = (double*)widthMat.data;
    double* ptrHeight = (double*)heightMat.data;
    for (int i = 0; i < length; i++)
    {
        ptrX[i] = history[i].normRect.x;
        ptrY[i] = history[i].normRect.y;
        ptrWidth[i] = history[i].normRect.width;
        ptrHeight[i] = history[i].normRect.height;
    }
    avgWidth = mean(widthMat)[0];
    avgHeight = mean(heightMat)[0];
    Scalar xMu, yMu, xSigma, ySigma;
    meanStdDev(xMat, xMu, xSigma);
    meanStdDev(yMat, yMu, ySigma);
    stdDevX = xSigma[0];
    stdDevY = ySigma[0];
}

static const double minSideLen = 20;
static const double maxSideLen = 0.8 * 320;
static const double minAspectRatio = 0.25;
static const double maxAspectRatio = 4;
static const double minMovStdDev = 3.25;
static const int minHistoryLen = 5;
static const int maxHistoryLen = 500;

namespace oip
{

void ObjectInfoParser::parse(const vector<zsfo::ObjectInfo>& src, vector<zpv::ObjectInfo>& dst)
{
    if (src.empty()) return;

    int size = src.size();
    dst.reserve(size);
    //fstream objectListFile;
    fstream trackletsFile;
    //objectListFile.open(listName.c_str(), ios::ate | ios::out | ios::in);
    trackletsFile.open(historyName.c_str(), ios::ate | ios::out | ios::in);
    for (int i = 0; i < size; i++)
    {
        const zsfo::ObjectInfo& refObj = src[i];
        if (!refObj.isFinal || !refObj.hasHistory || !refObj.hasVisualHistory) continue;

        double muWidth, muHeight, sigmaX, sigmaY;
        getNormHistoryProperties(refObj.history, muWidth, muHeight, sigmaX, sigmaY);
        double sideLen = max(muWidth, muHeight);
        double movStdDev = max(sigmaX, sigmaY);
        double aspectRatio = muWidth / muHeight;
        int historyLen = refObj.history.size();
        if (sideLen < minSideLen && movStdDev < minMovStdDev ||
            (aspectRatio < minAspectRatio || aspectRatio > maxAspectRatio) && historyLen < minHistoryLen ||
            sideLen > maxHistoryLen && historyLen > maxHistoryLen)
            continue;

        dst.push_back(zpv::ObjectInfo());
        zpv::ObjectInfo& procVideoObj = dst.back();
        procVideoObj.objectID = refObj.ID;
        procVideoObj.timeBegAndEnd.first = refObj.history.front().time;
        procVideoObj.timeBegAndEnd.second = refObj.history.back().time;
        ++objectCount;
        const zsfo::ObjectVisualRecord& refImage = refObj.visualHistory.front();
        procVideoObj.frameCount = refImage.number;
        procVideoObj.sliceLocation.x = refImage.rect.x;
        procVideoObj.sliceLocation.y = refImage.rect.y;
        procVideoObj.sliceLocation.width = refImage.rect.width;
        procVideoObj.sliceLocation.height = refImage.rect.height;
        string IDStr = getString(refObj.ID);
		string frameCountStr = getString(refImage.number);
        procVideoObj.sceneName = sceneNamePrefix + "ProcVideo_frame" + frameCountStr + ".jpg";
        procVideoObj.sliceName = sliceNamePrefix + "ProcVideo_frame" + frameCountStr + "_slice_" + IDStr + ".jpg";

        imwrite(procVideoObj.sceneName, refImage.scene);
        imwrite(procVideoObj.sliceName, refImage.slice);
        //objectListFile << setw(8) << refObj.ID
        //    << setw(12) << refImage.time << setw(12) << refImage.number
        //    << setw(8) << refImage.rect.x << setw(8) << refImage.rect.y
        //    << setw(8) << refImage.rect.width << setw(8) << refImage.rect.height;
        //objectListFile << "\n";
        const vector<zsfo::ObjectRecord>& refHistory = refObj.history;
		trackletsFile << "Object Count:  " << objectCount << "\n";
		trackletsFile << "ID:            " << refObj.ID << "\n";
		trackletsFile << "Size:          " << refHistory.size() << "\n";
		trackletsFile << "Frame Count Time Stamp       x       y       w       h" << "\n";
		for (int j = 0; j < refHistory.size(); j++)
		{
			trackletsFile << setw(11) << refHistory[j].number
				<< setw(11) << refHistory[j].time
				<< setw(8) << refHistory[j].normRect.x
				<< setw(8) << refHistory[j].normRect.y
				<< setw(8) << refHistory[j].normRect.width
				<< setw(8) << refHistory[j].normRect.height << "\n";
		}
		trackletsFile << "\n";
    }
    //objectListFile.close();
    trackletsFile.close();
}

void ObjectInfoParser::final(void)
{
    fstream file;
    //file.open(listName.c_str(), ios::ate | ios::out | ios::in);
	//file << "End\n";
	//file.close();
    file.open(historyName.c_str(), ios::ate | ios::out | ios::in);
	file << "End\n";
	file.close();
}

}

static string getNetFileName(const string& src)
{
    unsigned int posSlash = src.find_last_of("\\/");
    string nameWithExt = (posSlash == string::npos ? src : src.substr(posSlash + 1));
    unsigned int posDot = nameWithExt.find_last_of('.');
    return (posDot == string::npos ? nameWithExt : nameWithExt.substr(0, posDot));
}

static void pairToPoint(const vector<vector<pair<int, int> > >& pairs, vector<vector<Point> >& points)
{
    points.clear();
    if (pairs.empty()) return;
    int sizeSize = pairs.size();
    points.resize(sizeSize);
    for (int i = 0; i < sizeSize; i++)
    {
        int size = pairs[i].size();
        points[i].resize(size);
        for (int j = 0; j < size; j++)
            points[i][j] = Point(pairs[i][j].first, pairs[i][j].second);
    }
}

static void mul(vector<vector<Point> >& points, const ztool::Size2d& scale)
{
    if (points.empty()) return;
    int sizeSize = points.size();
    for (int i = 0; i < sizeSize; i++)
    {
        int size = points[i].size();
        for (int j = 0; j < size; j++)
            points[i][j] = ztool::mul(points[i][j], scale);
    }
}

namespace zpv
{

void procVideo(const TaskInfo& task, const ConfigInfo& config, 
    procVideoCallBack ptrCallBackFunc, void* ptrUserData)
{
    VideoCapture cap; 
    cap.open(task.videoPath);

    if (!cap.isOpened())
    {
        THROW_EXCEPT("cannot open " + task.videoPath);
    }

    double fps = cap.get(CV_CAP_PROP_FPS);
    int procEveryNFrame = (fps < 16) ? 1 : int(fps / 10 + 0.5);
	int totalFrameCount = cap.get(CV_CAP_PROP_FRAME_COUNT);

    int buildFrameCount = 50 * procEveryNFrame;
    int begIncCount = 0;
    if (task.frameCountBegAndEnd.first < buildFrameCount)
        buildFrameCount = 0;
    else
        begIncCount = task.frameCountBegAndEnd.first - buildFrameCount;
    int endIncCount = task.frameCountBegAndEnd.second;
        
    if (!cap.set(CV_CAP_PROP_POS_FRAMES, begIncCount))
    {
        THROW_EXCEPT("cannot locate frame count " + getString(begIncCount));
    }

    zsfo::StampedImage input;
    input.time = (long long int)cap.get(CV_CAP_PROP_POS_MSEC);
	input.number = (int)cap.get(CV_CAP_PROP_POS_FRAMES);
    cap.read(input.image);

    zsfo::MovingObjectDetector movObjDet;
    oip::ObjectInfoParser infoParser;

    Size origSize(input.image.size()), normSize(320, 240);
    int updateBackInterval = 2;
    int recordMode = zsfo::RecordMode::MultiVisualRecord;
    int saveMode = zsfo::SaveImageMode::SaveScene | zsfo::SaveImageMode::SaveSlice;
    int saveInterval = 2;
    int numOfSaved = 1;
    
    vector<vector<Point> > incPoints, excPoints;
    pairToPoint(config.includeRegion, incPoints);
    pairToPoint(config.excludeRegion, excPoints);
    ztool::Size2d scaleNormToOrig = ztool::div(normSize, origSize); 
    mul(incPoints, scaleNormToOrig);
    mul(excPoints, scaleNormToOrig);

    bool normScale = true;
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
        movObjDet.init(input, normSize, updateBackInterval, recordMode, saveMode, saveInterval, numOfSaved, 
            normScale, incPoints, excPoints, vector<Point>(),
            &minObjectArea, &minObjectWidth, &minObjectHeight, &charRegionCheck, charRegions,
            &checkTurnAround, &maxDistRectAndBlob, &minRatioIntersectToSelf, &minRatioIntersectToBlob);
		infoParser.init(task.saveImagePath, "", "", "", task.saveHistoryPath, task.historyFileName);
	}
	catch (const exception& e)
    {
        THROW_EXCEPT(e.what());
    }

    int progressInterval = 25;
    int procTotalCount = endIncCount - begIncCount + 1;
    for (int count = 1; count < procTotalCount; count++)
    {
        input.time = (long long int)cap.get(CV_CAP_PROP_POS_MSEC);
	    input.number = (int)cap.get(CV_CAP_PROP_POS_FRAMES);
        if (input.number >= totalFrameCount)
            break;
		if (!cap.read(input.image)) 
			continue;
        zsfo::ObjectDetails output;
        vector<ObjectInfo> objects;
        if (count % procEveryNFrame == 0)
        {
            try
		    {
                if (count < buildFrameCount)
                    movObjDet.build(input);
                else
                {
                    movObjDet.proc(input, output);
                    infoParser.parse(output.objects, objects);
                }
            }
            catch (const exception& e)
            {
                THROW_EXCEPT(e.what());
            }
        }
        if (ptrCallBackFunc && (count % progressInterval == 0 || !objects.empty()))
            ptrCallBackFunc(float(count) / procTotalCount * 100, objects, ptrUserData);
#if CMPL_SHOW_IMAGE        
		waitKey(output.objects.empty() ? 5 : 0);
#endif
    }

    zsfo::ObjectDetails output;
    vector<ObjectInfo> objects;
    movObjDet.final(output);
    infoParser.final();
    ptrCallBackFunc(100, objects, ptrUserData);
}

} // namespace zpv