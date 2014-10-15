#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ProcVideo.h"
#include "StringProcessor.h"
#include "Timer.h"

using namespace std;
using namespace cv;
using namespace zpv;
using namespace ztool;

void callBack(float progressPercentage, const vector<zpv::TaicangObjectInfo>& infos, void* ptrUserData)
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
    string videoPath = 
    /*"D:\\SHARED\\XinjiangVideo\\2014-08-31\\21.102.10.103_06_20140831193904656.mp4"*/
    "D:\\SHARED\\XinjiangVideo\\2014-08-30\\21.102.10.103_03_20140830182449984.mp4"
    /*"D:\\SHARED\\TrimpsVideo\\20140320\\Video09_2014_03_20_13_05_15_type1.avi"*/
    /*"D:\\SHARED\\TrimpsVideo\\20140320\\Video06_2014_03_20_13_00_15_type1.avi"*/
    /*"D:\\SHARED\\TrimpsVideo\\20140316\\01\\Video01_2014_03_16_14_57_28_type1.avi"*/
    /*"D:/SHARED/GuilinVideo/DVR1_灵川八里街川东一路八里一路路口20130909070538.avi"*/
    /*"D:\\SHARED\\MiscellaneousVideo\\DCS-7010L_20140416160622.avi"*/
    /*"D:\\SHARED\\MiscellaneousVideo\\案发前20秒(2).avi"*/
	/*"D:/SHARED/TaicangVideo/1/70.flv"*/
    /*"D:/SHARED/TaicangVideo/227省道、北门街_0_2014-08-20_10-09-08.asf"*/
    /*"D:/SHARED/TaicangVideo/209_32058517001310010090_40_2014-08-07 08_42_35~2014-08-07 08_47_35_bak.avi"*/
    /*"D:\\SHARED\\MiscellaneousVideo\\video\\4M2D12-21-2C.avi"*/
    /*"D:\\SHARED\\MiscellaneousVideo\\video\\test1030.avi"*/
    /*"D:\\SHARED\\MiscellaneousVideo\\video\\古北-头顶1.avi"*/
    /*"D:\\SHARED\\MiscellaneousVideo\\video\\顾戴2.avi"*/
    /*"D:\\SHARED\\MiscellaneousVideo\\养殖场130万像素.avi"*/
    /*"D:\\SHARED\\MiscellaneousVideo\\ch03_20140427105139MP4.avi"*/
    /*"D:\\SHARED\\MiscellaneousVideo\\05_29_08 06_03_11.avi"*/
    /*"D:/SHARED/TaicangVideo/CARCRASH_T112901_camera-t129f01_1_bak.avi"*/
    /*"D:/SHARED/TaicangVideo/2_M.dav"*/;

    string validVideoName;
    cvtPathToFileName(videoPath, validVideoName);
	TaicangTaskInfo task;
    TaicangParamInfo param;
	task.taskID = "0XFFFF";
	task.videoSegmentID = "0XABCD";
    task.videoPath = videoPath;
    task.saveImagePath = "result/" + validVideoName;
    task.saveHistoryPath = "result/" + validVideoName;
    task.historyFileName = "history.txt";
    param.normSize = make_pair(320, 240);
    param.normScale = true;
    param.minObjectArea = 50;
    param.minObjectWidth = 10;
    param.minObjectHeight = 10;
    param.maxMatchDist = 20;
    Timer timer;
    timer.start();
    try
    {
        procVideo(task, param, callBack, 0);
    }
    catch (const exception& e)
    {
        printf("%s\n", e.what());
    }
    timer.end();
    printf("time used is %.4f\nEND\n", timer.elapse());
    //system("pause");

    return 0;
}