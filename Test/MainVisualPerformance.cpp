#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "VisualInfo.h"
#include "BlobExtractor.h"
#include "Timer.h"

using namespace std;
using namespace cv;
using namespace zsfo;
using namespace ztool;

const static int normWidth = 352, normHeight = 288;
const static int procEveryNFrame = 4;
const static int updateEveryNProcFrame = 1;
const static double minWidth = 10, minHeight = 10, minArea = 100;

int main(void)
{
    const char* const path = /*"D:\\SHARED\\TrimpsVideo\\Video01_2013_10_23_17_50_55_type1.avi"*/
        /*"D:\\SHARED\\TrimpsVideo\\Video02_2013_10_23_17_50_55_type1.avi"*/
        /*"D:\\SHARED\\TrimpsVideo\\Video02_2013_10_23_18_10_02_type1.avi"*/
        /*"D:\\SHARED\\TrimpsVideo\\Video03_2013_10_23_17_50_55_type1.avi"*/
        /*"D:\\SHARED\\TrimpsVideo\\Video03_2013_10_23_18_00_58_type1.avi"*/
        /*"D:\\SHARED\\TrimpsVideo\\Video03_2013_10_23_18_10_02_type1.avi"*/
        /*"D:\\SHARED\\TrimpsVideo\\Video03_2013_10_23_18_30_10_type1.avi"*/
        /*"D:\\SHARED\\TrimpsVideo\\Video12_2013_10_24_07_00_10_type1.avi"*/
        /*"D:\\SHARED\\TrimpsVideo\\Video12_2013_10_24_07_10_10_type1.avi"*/
        /*"D:/SHARED/GuilinVideo/DVR1_灵川八里街川东一路八里一路路口_20130824184534_20130824185912_37633796_0001.avi"*/
        /*"D:/SHARED/GuilinVideo/DVR1_灵川八里街川东一路八里一路路口_20130825150550_20130825151929_60634656_0001.avi"*/
        /*"D:/SHARED/GuilinVideo/DVR1_灵川八里街川东一路八里一路路口_20130825160015_20130825161353_61618156_0001.avi"*/
        /*"D:/SHARED/GuilinVideo/DVR1_灵川八里街川东一路八里一路路口_20130825170814_20130825172152_62843812_0001.avi"*/
        /*"D:/SHARED/GuilinVideo/DVR1_灵川八里街川东一路八里一路路口_20130826170046_20130826171424_63094171_0001.avi"*/
        /*"D:/SHARED/GuilinVideo/DVR1_灵川八里街川东一路八里一路路口_20130827081157_20130827082536_54418000_0001.avi"*/
        /*"D:/SHARED/GuilinVideo/DVR1_灵川八里街川东一路八里一路路口20130909070538.avi"*/
        /*"D:/SHARED/TaicangVideo/1/70.flv"*/
        /*"D:\\SHARED\\TaicangVideo\\Night\\1.avi"*/
        "D:/SHARED/BishengRoadVideo/23711.flv";
    VideoCapture cap;
    cap.open(path);
    VisualInfo visualInfo;
    BlobExtractor blobExtractor;
    Size normSize(normWidth, normHeight);
    Timer totalTimer;
    RepeatTimer procTimer;
    int procCount = 1;
    int readCount = 0;
    int updateCount = 0;
    bool init = false;
    vector<Rect> noUpdate(1);
    noUpdate[0] = Rect(0, 0, normWidth, normHeight);
    vector<Rect> rects, stableRects;
    Mat frame, image;
    for (int count = 0; count < 5000; count++)
    {       
        if (!cap.read(frame)) break;
        if (readCount++ % procEveryNFrame)
            continue;
        resize(frame, image, normSize);
        GaussianBlur(image, image, Size(3, 3), 1.0);

        if (!init)
        {
            init = true;

            visualInfo.init(image);
            blobExtractor.init(normSize);
            blobExtractor.setConfigParams(&minArea, &minWidth, &minHeight);
            continue;
        }

        Mat foreImage;
        if (updateCount++ % updateEveryNProcFrame)
        {
            //printf("\n");
            procTimer.start();
            visualInfo.update(image, foreImage, false, noUpdate);
            procTimer.end();
        }
        else
        {
            //printf(", full update\n");
            procTimer.start();
            visualInfo.update(image, foreImage);
            procTimer.end();
        }
        //char buf[1024];
        //sprintf(buf, "D:\\SHARED\\GuilinVideo\\Result\\VisualInfoP4U1Thres145\\result%d.bmp", count);
        //imwrite(buf, foreImage);
        //imshow("image", image);
        //imshow("fore", foreImage);

        blobExtractor.proc(foreImage, Mat(), Mat(), rects, stableRects);
        int rectSize = rects.size();
        for (int i = 0; i < rectSize; i++)
            rectangle(image, rects[i], Scalar(0, 0, 255), 2);
        //imshow("blob image", image);
        //imshow("proc fore", foreImage);
        if (rectSize > 0)
        {
            char buf[1024];
            sprintf(buf, "D:\\SHARED\\BishengRoadVideo\\MOG\\result%d.bmp", count);
            Mat result(normHeight, normWidth * 2, CV_8UC3);
            Mat left = result(Rect(0, 0, normWidth, normHeight));
            image.copyTo(left);
            Mat right = result(Rect(normWidth, 0, normWidth, normHeight));
            Mat foreTemp(normHeight, normWidth, CV_8UC3);
            int fromTo[] = {0, 0, 0, 1, 0, 2};
            mixChannels(&foreImage, 1, &foreTemp, 1, fromTo, 3);
            foreTemp.copyTo(right);
            imwrite(buf, result);
        }

        //waitKey(30);
    }
    totalTimer.end();
    printf("avg proc time = %.6f, proc count = %d\n", procTimer.getAvgTime(), procTimer.getCount());
    printf("total proc time = %.6f\n", totalTimer.elapse());
    system("pause");
    return 0;
}