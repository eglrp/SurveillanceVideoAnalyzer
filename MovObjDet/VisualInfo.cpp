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

static void filterSmallRegions(cv::Mat& mask, int minWidth, int minHeight, int minArea)
{
    int width = mask.cols;
    int height = mask.rows;
    cv::Mat scan;
    scan.create(height, width, CV_8UC1);
    scan.setTo(0);
    cv::Mat comp;
    comp.create(height, width, CV_8UC1);
    comp.setTo(0);
    std::vector<unsigned char*> ptrMaskRow(height);
    unsigned char **ptrMask = &ptrMaskRow[0];
    std::vector<unsigned char*> ptrScanRow(height);
    unsigned char **ptrScan = &ptrScanRow[0];
    std::vector<unsigned char*> ptrCompRow(height);
    unsigned char **ptrComp = &ptrCompRow[0];
    for (int i = 0; i < height; i++)
    {
        ptrMask[i] = mask.ptr<unsigned char>(i);
        ptrScan[i] = scan.ptr<unsigned char>(i);
        ptrComp[i] = comp.ptr<unsigned char>(i);
    } 
    std::vector<cv::Point> stack;    
    stack.resize(height * width);
    cv::Point* ptrStack = &stack[0];
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int stackSize = 0, area = 0;
            if (ptrScan[i][j] == 0X00 && ptrMask[i][j] == 0XFF)
            {                
                ptrStack[stackSize] = cv::Point(j, i);
                stackSize++;
                ptrScan[i][j] = 0XFF;
                int minX = j, minY = i, maxX = j, maxY = i; 
                while (stackSize)
                {
                    area++;
                    stackSize--;
                    int x = ptrStack[stackSize].x, y = ptrStack[stackSize].y;                    
                    ptrComp[y][x] = 0XFF;
                    minX = std::min(minX, x);
                    minY = std::min(minY, y);
                    maxX = std::max(maxX, x);                    
                    maxY = std::max(maxY, y);
                    for (int offy = -1; offy <= 1; offy++)
                    {
                        for (int offx = -1; offx <= 1; offx++)
                        {                            
                            int currX = x + offx, currY = y + offy;
                            if (currX >= 0 && currX < width &&
                                currY >= 0 && currY < height &&
                                ptrMask[currY][currX] == 0XFF && ptrScan[currY][currX] == 0X00)
                            {
                                ptrStack[stackSize] = cv::Point(currX, currY);
                                stackSize++;
                                ptrScan[currY][currX] = 0XFF;
                            }
                        }
                    }                    
                }
                cv::Rect currRect(minX, minY, maxX - minX + 1, maxY - minY + 1);
                if (currRect.width <= minWidth && currRect.height <= minHeight || area <= minArea ||
                    area < minWidth * currRect.height || area < minHeight * currRect.width)
                {
                    for (int u = currRect.y; u < currRect.y + currRect.height; u++)
                    {
                        for (int v = currRect.x; v < currRect.x + currRect.width; v++)
                        {
                            if (ptrComp[u][v])
                                ptrMask[u][v] = 0;
                        }
                    }
                }
                comp(currRect).setTo(0);
            }
        }
    }
}

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
    blur(normGrayImage, normGrayImage, Size(3, 3));
    calcThresholdedGradient(normGrayImage, normGradImage, 145);//145
    if (fullUpdate) 
        calcThresholdedGradient(backGrayImage, backGradImage, 145);//145
    normGradImage.copyTo(gradDiffImage);
    gradDiffImage.setTo(0, backGradImage);
#if CMPL_SHOW_IMAGE
    imshow("Gradient Diff", gradDiffImage);
#endif
    medianBlur(gradDiffImage, gradDiffImage, 3);
#if CMPL_SHOW_IMAGE
    imshow("Frame Gradient", normGradImage);
    imshow("Back Frame Gradient", backGradImage);
    imshow("Blurred Gradient Diff", gradDiffImage);
#endif

    //filterSmallRegions(gradDiffImage, 5, 5, 20);
#if CMPL_SHOW_IMAGE
    imshow("Filtered Blurred Gradient Diff", gradDiffImage);
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

