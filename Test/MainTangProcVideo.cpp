﻿#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ProcVideo.h"
#include "Timer.h"

using namespace std;
using namespace cv;
using namespace zpv;
using namespace ztool;

void callBack(float progressPercentage, const vector<zpv::ObjectInfo>& infos, void* ptrUserData)
{    
	if (infos.empty())
	{
		printf("process: %.4f%%\n\n", progressPercentage);
		return;
	}

	printf("process: %.4f%%, has object(s)\n", progressPercentage);
	int size = infos.size();
	for (int i = 0; i < size; i++)
	{
		printf("  Obj ID = %d\n", infos[i].objectID);
		printf("  time stamp = (%lld, %lld)\n", infos[i].timeBegAndEnd.first, infos[i].timeBegAndEnd.second);
		printf("  scene path = %s\n", infos[i].sceneName.c_str());
		printf("  slice path = %s\n", infos[i].sliceName.c_str());
		printf("  slice location = (%d, %d, %d, %d)\n", 
			infos[i].sliceLocation.x, infos[i].sliceLocation.y,
			infos[i].sliceLocation.width, infos[i].sliceLocation.height);
		printf("\n");
	}
}

int main(int argc, char** argv)
{
    string videoPath = /*"D:\\SHARED\\TrimpsVideo\\20140320\\Video09_2014_03_20_13_05_15_type1.avi"*/
    /*"D:\\SHARED\\TrimpsVideo\\20140320\\Video06_2014_03_20_13_00_15_type1.avi"*/
    /*"D:\\SHARED\\TrimpsVideo\\20140316\\01\\Video01_2014_03_16_14_57_28_type1.avi"*/
    /*"D:/SHARED/GuilinVideo/DVR1_灵川八里街川东一路八里一路路口20130909070538.avi"*/
	"D:/SHARED/TaicangVideo/1/70.flv";
	VideoCapture cap;
	cap.open(videoPath);
	int totalFrameCount = cap.get(CV_CAP_PROP_FRAME_COUNT);
	//printf("frame count %d\n", totalFrameCount);
	//Timer timer;
	//while (true)
	//{
	//	int frame = cap.get(CV_CAP_PROP_POS_FRAMES);
	//	Mat img;
	//	if (!cap.read(img))
	//	{
	//		printf("cannot read frame %d\n", frame);
	//		break;
	//	}
	//}
	//timer.end();
	//cap.release();
	//printf("time = %.4f\n", timer.elapse());
	//system("pause");
	//return 0;
	//system("pause");

	TaskInfo task;
    ConfigInfo config;
	task.taskID = "0XFFFF";
	task.videoSegmentID = "0XABCD";
    task.videoPath = videoPath;
    task.saveImagePath = "result/temp";
    task.saveHistoryPath = "result/temp";
    task.historyFileName = "history.txt";
    task.frameCountBegAndEnd = make_pair(0, totalFrameCount - 1);
	config.tiltType = TiltType::SMALL;
	config.zoomType = ZoomType::NEAR;
	config.environmentType = EnvironmentType::SUNNY;
    Timer timer;
    timer.start();
    try
    {
        procVideo(task, config, callBack, 0);
    }
    catch (const exception& e)
    {
        printf("%s\n", e.what());
    }
    timer.end();
    printf("time used is %.4f\n", timer.elapse());
    system("pause");

    return 0;
}