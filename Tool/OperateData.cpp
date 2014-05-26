#include <opencv2/imgproc/imgproc.hpp>
#include "OperateData.h"
#include "Exception.h"
using namespace std;
using namespace cv;

namespace ztool
{

Scalar calcAvgAbsDiff(const Mat& src1, const Mat& src2, const Rect& currRect)
{
    if (src1.data == 0 || src2.data == 0)
		THROW_EXCEPT("Mat::data = 0");

	if (src1.type() != CV_8UC3 || src2.type() != CV_8UC3)
		THROW_EXCEPT("unsupported element type");

	if (src1.rows != src2.rows || src1.cols != src2.cols)
		THROW_EXCEPT("src1 and src2 do not share the same size");

    int accAbsDiffB = 0, accAbsDiffG = 0, accAbsDiffR = 0;
    int accSrc1B = 0, accSrc1G = 0, accSrc1R = 0;
    int accSrc2B = 0, accSrc2G = 0, accSrc2R = 0;
    for (int i = currRect.y, iEnd = currRect.y + currRect.height; i < iEnd; i++)
    {
        const unsigned char* ptrData1 = src1.ptr<unsigned char>(i);
        const unsigned char* ptrData2 = src2.ptr<unsigned char>(i);
        for (int j = currRect.x, jEnd = currRect.x + currRect.width; j < jEnd; j++)
        {
            accAbsDiffB += abs(ptrData1[j * 3] - ptrData2[j * 3]);
            accAbsDiffG += abs(ptrData1[j * 3 + 1] - ptrData2[j * 3 + 1]);
            accAbsDiffR += abs(ptrData1[j * 3 + 2] - ptrData2[j * 3 + 2]);
            accSrc1B += ptrData1[j * 3];
            accSrc1G += ptrData1[j * 3 + 1];
            accSrc1R += ptrData1[j * 3 + 2];
            accSrc2B += ptrData2[j * 3];
            accSrc2G += ptrData2[j * 3 + 1];
            accSrc2R += ptrData2[j * 3 + 2];
        }
    }
    double pixelCount = currRect.width * currRect.height;
    return Scalar(accAbsDiffB / sqrt(double(accSrc1B) * double(accSrc2B) + 0.0001), 
                  accAbsDiffG / sqrt(double(accSrc1G) * double(accSrc2G) + 0.0001), 
                  accAbsDiffR / sqrt(double(accSrc1R) * double(accSrc2R) + 0.0001));
}

Scalar calcCenterCorrRatio(const Mat& src1, const Mat& src2, const Rect& currRect)
{
	if (src1.data == 0 || src2.data == 0)
		THROW_EXCEPT("Mat::data = 0");

	if (src1.type() != CV_8UC3 || src2.type() != CV_8UC3)
		THROW_EXCEPT("unsupported element type");

	if (src1.rows != src2.rows || src1.cols != src2.cols)
		THROW_EXCEPT("src1 and src2 do not share the same size");

    int mean1B = 0, mean1G = 0, mean1R = 0;
    int mean2B = 0, mean2G = 0, mean2R = 0;
    for (int i = currRect.y, iEnd = currRect.y + currRect.height; i < iEnd; i++)
    {
        const unsigned char* ptrData1 = src1.ptr<unsigned char>(i);
        const unsigned char* ptrData2 = src2.ptr<unsigned char>(i);
        for (int j = currRect.x, jEnd = currRect.x + currRect.width; j < jEnd; j++)
        {
            mean1B += ptrData1[j * 3];
            mean1G += ptrData1[j * 3 + 1];
            mean1R += ptrData1[j * 3 + 2];
            mean2B += ptrData2[j * 3];
            mean2G += ptrData2[j * 3 + 1];
            mean2R += ptrData2[j * 3 + 2];
        }
    }
    double pixelCount = currRect.width * currRect.height;
    double meanSrc1B = mean1B / pixelCount;
    double meanSrc1G = mean1G / pixelCount;
    double meanSrc1R = mean1R / pixelCount;
    double meanSrc2B = mean2B / pixelCount;
    double meanSrc2G = mean2G / pixelCount;
    double meanSrc2R = mean2R / pixelCount;

    double covB = 0.0, covG = 0.0, covR = 0.0;
    double varSrc1B = 0.0, varSrc1G = 0.0, varSrc1R = 0.0;
    double varSrc2B = 0.0, varSrc2G = 0.0, varSrc2R = 0.0;
    for (int i = currRect.y; i < currRect.y + currRect.height; i++)
    {
        const unsigned char* ptrData1 = src1.ptr<unsigned char>(i);
        const unsigned char* ptrData2 = src2.ptr<unsigned char>(i);
        for (int j = currRect.x, jEnd = currRect.x + currRect.width; j < jEnd; j++)
        {
            double diff1B = double(ptrData1[j * 3]) - meanSrc1B;
            double diff1G = double(ptrData1[j * 3 + 1]) - meanSrc1G;
            double diff1R = double(ptrData1[j * 3 + 2]) - meanSrc1R;
            double diff2B = double(ptrData2[j * 3]) - meanSrc2B;
            double diff2G = double(ptrData2[j * 3 + 1]) - meanSrc2G;
            double diff2R = double(ptrData2[j * 3 + 2]) - meanSrc2R;
            covB += diff1B * diff2B;
            covG += diff1G * diff2G;
            covR += diff1R * diff2R;
            varSrc1B += diff1B * diff1B;
            varSrc1G += diff1G * diff1G;
            varSrc1R += diff1R * diff1R;
            varSrc2B += diff2B * diff2B;
            varSrc2G += diff2G * diff2G;
            varSrc2R += diff2R * diff2R;

            /*covB += (float(ptrData1[j * 3]) - meanSrc1B) * (float(ptrData2[j * 3]) - meanSrc2B);
            covG += (float(ptrData1[j * 3 + 1]) - meanSrc1G) * (float(ptrData2[j * 3 + 1]) - meanSrc2G);
            covR += (float(ptrData1[j * 3 + 2]) - meanSrc1R) * (float(ptrData2[j * 3 + 2]) - meanSrc2R);
            varSrc1B += pow(float(ptrData1[j * 3]) - meanSrc1B, 2);
            varSrc1G += pow(float(ptrData1[j * 3 + 1]) - meanSrc1G, 2);
            varSrc1R += pow(float(ptrData1[j * 3 + 2]) - meanSrc1R, 2);
            varSrc2B += pow(float(ptrData2[j * 3]) - meanSrc2B, 2);
            varSrc2G += pow(float(ptrData2[j * 3 + 1]) - meanSrc2G, 2);
            varSrc2R += pow(float(ptrData2[j * 3 + 2]) - meanSrc2R, 2);*/
        }
    }

    Scalar corrRatio;
    corrRatio[0] = covB / sqrt(varSrc1B * varSrc2B + 0.0001);
    corrRatio[1] = covG / sqrt(varSrc1G * varSrc2G + 0.0001);
    corrRatio[2] = covR / sqrt(varSrc1R * varSrc2R + 0.0001);
    return corrRatio;
}

Scalar calcCenterCorrRatio(const Mat& src1, const Mat& src2, const Rect& currRect, const Mat& maskImage)
{
	if(src1.data == 0 || src2.data == 0 || maskImage.data == 0)
		THROW_EXCEPT("Mat::data = 0");

	if (src1.type() != CV_8UC3 || src2.type() != CV_8UC3)
		THROW_EXCEPT("unsupported element type");

	if (src1.rows != src2.rows || src1.cols != src2.cols)
		THROW_EXCEPT("src1 and src2 do not share the same size");

    int count = 0;

    int mean1B = 0, mean1G = 0, mean1R = 0;
    int mean2B = 0, mean2G = 0, mean2R = 0;
    for (int i = currRect.y, iEnd = currRect.y + currRect.height; i < iEnd; i++)
    {
        const unsigned char* ptrData1 = src1.ptr<unsigned char>(i);
        const unsigned char* ptrData2 = src2.ptr<unsigned char>(i);
        const unsigned char* ptrMaskData = maskImage.ptr<unsigned char>(i);
        for (int j = currRect.x, jEnd = currRect.x + currRect.width; j < jEnd; j++)
        {
            if (ptrMaskData[j] == 0)
                continue;
            count++;
            mean1B += ptrData1[j * 3];
            mean1G += ptrData1[j * 3 + 1];
            mean1R += ptrData1[j * 3 + 2];
            mean2B += ptrData2[j * 3];
            mean2G += ptrData2[j * 3 + 1];
            mean2R += ptrData2[j * 3 + 2];
        }
    }
    double pixelCount = count;
    double meanSrc1B = mean1B / pixelCount;
    double meanSrc1G = mean1G / pixelCount;
    double meanSrc1R = mean1R / pixelCount;
    double meanSrc2B = mean2B / pixelCount;
    double meanSrc2G = mean2G / pixelCount;
    double meanSrc2R = mean2R / pixelCount;

    double covB = 0.0, covG = 0.0, covR = 0.0;
    double varSrc1B = 0.0, varSrc1G = 0.0, varSrc1R = 0.0;
    double varSrc2B = 0.0, varSrc2G = 0.0, varSrc2R = 0.0;
    for (int i = currRect.y; i < currRect.y + currRect.height; i++)
    {
        const unsigned char* ptrData1 = src1.ptr<unsigned char>(i);
        const unsigned char* ptrData2 = src2.ptr<unsigned char>(i);
        const unsigned char* ptrMaskData = maskImage.ptr<unsigned char>(i);
        for (int j = currRect.x; j < currRect.x + currRect.width; j++)
        {
            if (ptrMaskData[j] == 0)
                continue;

            double diff1B = double(ptrData1[j * 3]) - meanSrc1B;
            double diff1G = double(ptrData1[j * 3 + 1]) - meanSrc1G;
            double diff1R = double(ptrData1[j * 3 + 2]) - meanSrc1R;
            double diff2B = double(ptrData2[j * 3]) - meanSrc2B;
            double diff2G = double(ptrData2[j * 3 + 1]) - meanSrc2G;
            double diff2R = double(ptrData2[j * 3 + 2]) - meanSrc2R;
            covB += diff1B * diff2B;
            covG += diff1G * diff2G;
            covR += diff1R * diff2R;
            varSrc1B += diff1B * diff1B;
            varSrc1G += diff1G * diff1G;
            varSrc1R += diff1R * diff1R;
            varSrc2B += diff2B * diff2B;
            varSrc2G += diff2G * diff2G;
            varSrc2R += diff2R * diff2R;

            /*covB += (float(ptrData1[j * 3]) - meanSrc1B) * (float(ptrData2[j * 3]) - meanSrc2B);
            covG += (float(ptrData1[j * 3 + 1]) - meanSrc1G) * (float(ptrData2[j * 3 + 1]) - meanSrc2G);
            covR += (float(ptrData1[j * 3 + 2]) - meanSrc1R) * (float(ptrData2[j * 3 + 2]) - meanSrc2R);
            varSrc1B += pow(float(ptrData1[j * 3]) - meanSrc1B, 2);
            varSrc1G += pow(float(ptrData1[j * 3 + 1]) - meanSrc1G, 2);
            varSrc1R += pow(float(ptrData1[j * 3 + 2]) - meanSrc1R, 2);
            varSrc2B += pow(float(ptrData2[j * 3]) - meanSrc2B, 2);
            varSrc2G += pow(float(ptrData2[j * 3 + 1]) - meanSrc2G, 2);
            varSrc2R += pow(float(ptrData2[j * 3 + 2]) - meanSrc2R, 2);*/
        }
    }

    Scalar corrRatio;
    corrRatio[0] = covB / sqrt(varSrc1B * varSrc2B + 0.0001);
    corrRatio[1] = covG / sqrt(varSrc1G * varSrc2G + 0.0001);
    corrRatio[2] = covR / sqrt(varSrc1R * varSrc2R + 0.0001);
    return corrRatio;
}

Scalar calcOriginCorrRatio(const Mat& src1, const Mat& src2, const Rect& currRect)
{
	if (src1.data == 0 || src2.data == 0)
		THROW_EXCEPT("Mat::data = 0");

	if (src1.type() != CV_8UC3 || src2.type() != CV_8UC3)
		THROW_EXCEPT("unsupported element type");

	if (src1.rows != src2.rows || src1.cols != src2.cols)
		THROW_EXCEPT("src1 and src2 do not share the same size");

    double covB = 0.0, covG = 0.0, covR = 0.0;
    double varSrc1B = 0.0, varSrc1G = 0.0, varSrc1R = 0.0;
    double varSrc2B = 0.0, varSrc2G = 0.0, varSrc2R = 0.0;
    for (int i = currRect.y, iEnd = currRect.y + currRect.height; i < iEnd; i++)
    {
        const unsigned char* ptrData1 = src1.ptr<unsigned char>(i);
        const unsigned char* ptrData2 = src2.ptr<unsigned char>(i);
        for (int j = currRect.x, jEnd = currRect.x + currRect.width; j < jEnd; j++)
        {            
            int b1 = ptrData1[j * 3], g1 = ptrData1[j * 3 + 1], r1 = ptrData1[j * 3 + 2];
            int b2 = ptrData2[j * 3], g2 = ptrData2[j * 3 + 1], r2 = ptrData2[j * 3 + 2];
            covB += b1 * b2;
            covG += g1 * g2;
            covR += r1 * r2;
            varSrc1B += b1 * b1;
            varSrc1G += g1 * g1;
            varSrc1R += r1 * r1;
            varSrc2B += b2 * b2;
            varSrc2G += g2 * g2;
            varSrc2R += r2 * r2;

            /*covB += double(ptrData1[j * 3]) * double(ptrData2[j * 3]);
            covG += double(ptrData1[j * 3 + 1]) * double(ptrData2[j * 3 + 1]);
            covR += double(ptrData1[j * 3 + 2]) * double(ptrData2[j * 3 + 2]);
            varSrc1B += pow(double(ptrData1[j * 3]), 2);
            varSrc1G += pow(double(ptrData1[j * 3 + 1]), 2);
            varSrc1R += pow(double(ptrData1[j * 3 + 2]), 2);
            varSrc2B += pow(double(ptrData2[j * 3]), 2);
            varSrc2G += pow(double(ptrData2[j * 3 + 1]), 2);
            varSrc2R += pow(double(ptrData2[j * 3 + 2]), 2);*/
        }
    }

    Scalar corrRatio;
	corrRatio[0] = covB / sqrt(varSrc1B * varSrc2B + 0.0001);
    corrRatio[1] = covG / sqrt(varSrc1G * varSrc2G + 0.0001);
    corrRatio[2] = covR / sqrt(varSrc1R * varSrc2R + 0.0001);
    return corrRatio;
}

Scalar calcOriginCorrRatio(const Mat& src1, const Mat& src2, const Rect& currRect, const Mat& maskImage)
{
	if (src1.data == 0 || src2.data == 0 || maskImage.data == 0)
		THROW_EXCEPT("Mat::data = 0");

	if (src1.type() != CV_8UC3 || src2.type() != CV_8UC3)
		THROW_EXCEPT("unsupported element type");

	if (src1.rows != src2.rows || src1.cols != src2.cols)
		THROW_EXCEPT("src1 and src2 do not share the same size");

    double covB = 0.0, covG = 0.0, covR = 0.0;
    double varSrc1B = 0.0, varSrc1G = 0.0, varSrc1R = 0.0;
    double varSrc2B = 0.0, varSrc2G = 0.0, varSrc2R = 0.0;
    for (int i = currRect.y, iEnd = currRect.y + currRect.height; i < iEnd; i++)
    {
        const unsigned char* ptrData1 = src1.ptr<unsigned char>(i);
        const unsigned char* ptrData2 = src2.ptr<unsigned char>(i);
        const unsigned char* ptrMaskData = maskImage.ptr<unsigned char>(i);
        for (int j = currRect.x; j < currRect.x + currRect.width; j++)
        {
            if (ptrMaskData[j] == 0)
                continue;
            int b1 = ptrData1[j * 3], g1 = ptrData1[j * 3 + 1], r1 = ptrData1[j * 3 + 2];
            int b2 = ptrData2[j * 3], g2 = ptrData2[j * 3 + 1], r2 = ptrData2[j * 3 + 2];
            covB += b1 * b2;
            covG += g1 * g2;
            covR += r1 * r2;
            varSrc1B += b1 * b1;
            varSrc1G += g1 * g1;
            varSrc1R += r1 * r1;
            varSrc2B += b2 * b2;
            varSrc2G += g2 * g2;
            varSrc2R += r2 * r2;

            /*covB += double(ptrData1[j * 3]) * double(ptrData2[j * 3]);
            covG += double(ptrData1[j * 3 + 1]) * double(ptrData2[j * 3 + 1]);
            covR += double(ptrData1[j * 3 + 2]) * double(ptrData2[j * 3 + 2]);
            varSrc1B += pow(double(ptrData1[j * 3]), 2);
            varSrc1G += pow(double(ptrData1[j * 3 + 1]), 2);
            varSrc1R += pow(double(ptrData1[j * 3 + 2]), 2);
            varSrc2B += pow(double(ptrData2[j * 3]), 2);
            varSrc2G += pow(double(ptrData2[j * 3 + 1]), 2);
            varSrc2R += pow(double(ptrData2[j * 3 + 2]), 2);*/
        }
    }

    Scalar corrRatio;
	corrRatio[0] = covB / sqrt(varSrc1B * varSrc2B + 0.0001);
    corrRatio[1] = covG / sqrt(varSrc1G * varSrc2G + 0.0001);
    corrRatio[2] = covR / sqrt(varSrc1R * varSrc2R + 0.0001);
    return corrRatio;
}

void calcElemWiseL1Norm(const Mat& src1, const Mat& src2, Mat& dst)
{
	if (src1.data == 0 || src2.data == 0)
		THROW_EXCEPT("Mat::data = 0");

	if (src1.type() != CV_32FC1 || src2.type() != CV_32FC1)
		THROW_EXCEPT("unsupported element type");

	if (src1.rows != src2.rows || src1.cols != src2.cols)
		THROW_EXCEPT("src1 and src2 do not share the same size");

	//dst = abs(src1) + abs(src2);
    dst.create(src1.rows, src1.cols, CV_32FC1);
    for (int i = 0; i < dst.rows; i++)
    {
        const float* ptrSrc1Data = src1.ptr<float>(i);
        const float* ptrSrc2Data = src2.ptr<float>(i);
        float* ptrDstData = dst.ptr<float>(i);
        for (int j = 0; j < dst.cols; j++)
        {
            ptrDstData[j] = fabs(ptrSrc1Data[j]) + fabs(ptrSrc2Data[j]);
        }
    }
}

void calcElemWiseL2Norm(const Mat& src1, const Mat& src2, Mat& dst)
{
	if (src1.data == 0 || src2.data == 0)
		THROW_EXCEPT("Mat::data = 0");

	if (src1.type() != CV_32FC1 || src2.type() != CV_32FC1)
		THROW_EXCEPT("unsupported element type");

	if (src1.rows != src2.rows || src1.cols != src2.cols)
		THROW_EXCEPT("src1 and src2 do not share the same size");

	//Mat sqr1, sqr2;
    //pow(src1, 2, sqr1);
    //pow(src2, 2, sqr2);
    //sqrt(sqr1 + sqr2, dst);
    dst.create(src1.rows, src1.cols, CV_32FC1);
    for (int i = 0; i < dst.rows; i++)
    {
        const float* ptrSrc1Data = src1.ptr<float>(i);
        const float* ptrSrc2Data = src2.ptr<float>(i);
        float* ptrDstData = dst.ptr<float>(i);
        for (int j = 0; j < dst.cols; j++)
        {
            ptrDstData[j] = sqrt(pow(ptrSrc1Data[j], 2) + pow(ptrSrc2Data[j], 2));
        }
    }
}

}

//static float horiArray[3][3] = {{1, 0, -1}, {sqrt(2.0F), 0, -sqrt(2.0F)}, {1, 0, -1}};
//static float vertArray[3][3] = {{1, sqrt(2.0F), 1}, {0, 0, 0}, {-1, -sqrt(2.0F), -1}};
static float horiArray[3][3] = {{3, 0, -3}, {10, 0, -10}, {3, 0, -3}};
static float vertArray[3][3] = {{3, 10, 3}, {0, 0, 0}, {-3, -10, -3}};
static Mat horiKernel = Mat(3, 3, CV_32F, horiArray);
static Mat vertKernel = Mat(3, 3, CV_32F, vertArray);

namespace ztool
{

void calcThresholdedGradient(const Mat& src, Mat& dst, double thres)
{
	if(src.data == 0)
		THROW_EXCEPT("Mat::data = 0");

	if (src.type() != CV_8UC1)
		THROW_EXCEPT("unsupported element type");

    Mat horiGrad = Mat::zeros(src.rows, src.cols, CV_32FC1);
    Mat vertGrad = Mat::zeros(src.rows, src.cols, CV_32FC1);
	Mat grad = Mat::zeros(src.rows, src.cols, CV_32FC1);

    filter2D(src, horiGrad, horiGrad.depth(), horiKernel);
    filter2D(src, vertGrad, vertGrad.depth(), vertKernel);
    calcElemWiseL1Norm(horiGrad, vertGrad, grad);
	grad.convertTo(dst, CV_8UC1); 

    //threshold(dst, dst, thres, 255, THRESH_BINARY);
    //dst.row(0) = 0;
    //dst.row(dst.rows - 1) = 0;
    //dst.col(0) = 0;
    //dst.col(dst.cols - 1) = 0;
    /*unsigned char* ptrDstData = (unsigned char*)dst.data;
    for (int i = 0; i < dst.rows; i++)
    {
        ptrDstData = dst.ptr<unsigned char>(i);
        ptrDstData[0] = 0;
        ptrDstData[dst.cols - 1] = 0;
    }
    ptrDstData = dst.ptr<unsigned char>(0);
    for (int i = 0; i < dst.cols; i++)
    {
        ptrDstData[i] = 0;
    }
    ptrDstData = dst.ptr<unsigned char>(dst.rows - 1);
    for (int i = 0; i < dst.cols; i++)
    {
        ptrDstData[i] = 0;
    }*/
    for (int i = 0; i < dst.rows; i++)
    {
        unsigned char* ptrDstData = dst.ptr<unsigned char>(i);
        for (int j = 0; j < dst.cols; j++)
        {
            if (ptrDstData[j] > thres)
                ptrDstData[j] = 255;
            else
                ptrDstData[j] = 0;
        }
    }
}

void calcGradient(const Mat& src, Mat& dst, double scale)
{
	if (src.data == 0)
		THROW_EXCEPT("Mat::data = 0");

	if (src.type() != CV_8UC1)
		THROW_EXCEPT("unsupported element type");
	
	Mat horiGrad = Mat::zeros(src.rows, src.cols, CV_32FC1);
    Mat vertGrad = Mat::zeros(src.rows, src.cols, CV_32FC1);
	Mat grad = Mat::zeros(src.rows, src.cols, CV_32FC1);

    filter2D(src, horiGrad, horiGrad.depth(), horiKernel);
    filter2D(src, vertGrad, vertGrad.depth(), vertKernel);
    calcElemWiseL1Norm(horiGrad, vertGrad, grad);
	if (scale != 1.0) grad *= scale;
	grad.convertTo(dst, CV_8UC1); 

    //dst.row(0) = 0;
    //dst.row(dst.rows - 1) = 0;
    //dst.col(0) = 0;
    //dst.col(dst.cols - 1) = 0;
    /*unsigned char* ptrDstData = (unsigned char*)dst.data;
    for (int i = 0; i < dst.rows; i++)
    {
        ptrDstData = dst.ptr<unsigned char>(i);
        ptrDstData[0] = 0;
        ptrDstData[dst.cols - 1] = 0;
    }
    ptrDstData = dst.ptr<unsigned char>(0);
    for (int i = 0; i < dst.cols; i++)
    {
        ptrDstData[i] = 0;
    }
    ptrDstData = dst.ptr<unsigned char>(dst.rows - 1);
    for (int i = 0; i < dst.cols; i++)
    {
        ptrDstData[i] = 0;
    }*/
}

}