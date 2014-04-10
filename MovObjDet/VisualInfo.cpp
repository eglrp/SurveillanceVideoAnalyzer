#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "VisualInfo.h"
#include "ExtendedMog.h"
#include "CompileControl.h"
#include "OperateData.h"
#include "ShowData.h"

using namespace std;
using namespace cv;
using namespace ztool;

namespace zsfo
{

void VisualInfo::init(const Mat& image)
{
	width = image.cols;
	height = image.rows;
	normGrayImage = Mat(height, width, CV_8UC1);
	backGrayImage = Mat(height, width, CV_8UC1);
	normGradImage = Mat(height, width, CV_8UC1);
    backGradImage = Mat(height, width, CV_8UC1);

	// 初始化背景模型
	backModel = new Mog;
	backModel->init(image);
}

void VisualInfo::update(const Mat& image, 
	bool fullUpdate, const vector<Rect>& rectsNoUpdate)
{
	Mat foreImage, backImage, gradDiffImage;
	update(image, foreImage, backImage, gradDiffImage, fullUpdate, rectsNoUpdate);
}

void VisualInfo::update(const Mat& image, Mat& foreImage, 
	bool fullUpdate, const vector<Rect>& rectsNoUpdate)
{
	Mat backImage, gradDiffImage;
	update(image, foreImage, backImage, gradDiffImage, fullUpdate, rectsNoUpdate);
}

void VisualInfo::update(const Mat& image, Mat& foreImage, Mat& backImage, 
	bool fullUpdate, const vector<Rect>& rectsNoUpdate)
{
	Mat gradDiffImage;
	update(image, foreImage, backImage, gradDiffImage, fullUpdate, rectsNoUpdate);
}

void VisualInfo::update(const Mat& image, Mat& foreImage, Mat& backImage, Mat& gradDiffImage, 
	bool fullUpdate, const vector<Rect>& rectsNoUpdate)
{
	// 更新背景模型
	if (fullUpdate)
        backModel->update(image, foreImage, backImage, rectsNoUpdate);
    else
    {
        vector<Rect> rects(1);
        rects[0] = Rect(0, 0, width, height);
        backModel->update(image, foreImage, backImage, rects);
    }
#if CMPL_SHOW_IMAGE
	imshow("background image", backImage);
	imshow("foreground image", foreImage);
#endif

	// 将归一化帧和背景帧转换为灰度帧
    cvtColor(image, normGrayImage, CV_BGR2GRAY);
    if (fullUpdate) 
		cvtColor(backImage, backGrayImage, CV_BGR2GRAY);

    // 计算梯度差
	calcThresholdedGradient(normGrayImage, normGradImage, 145);//145
    if (fullUpdate) 
		calcThresholdedGradient(backGrayImage, backGradImage, 145);//145
	normGradImage.copyTo(gradDiffImage);
	gradDiffImage.setTo(0, backGradImage);
    medianBlur(gradDiffImage, gradDiffImage, 3);
#if CMPL_SHOW_IMAGE
    imshow("Frame Gradient", normGradImage);
    imshow("Back Frame Gradient", backGradImage);
    imshow("Gradient Diff", gradDiffImage);
#endif

    // 给前景图加上梯度差值
	for (int i = 0; i < foreImage.rows; i++)
    {
        unsigned char* ptrForeData = foreImage.ptr<unsigned char>(i);
		unsigned char* ptrGradData = gradDiffImage.ptr<unsigned char>(i);
        for (int j = 0; j < foreImage.cols; j++)
        {
            if (ptrGradData[j] == 0XFF)
                ptrForeData[j] = 0XFF;
        }
    }
#if CMPL_SHOW_IMAGE
    imshow("Foreground After Add Edge", foreImage);
#endif
}

}

