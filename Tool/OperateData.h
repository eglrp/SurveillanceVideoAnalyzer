#pragma once

#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>

namespace ztool
{

// 计算像素差的平均值
cv::Scalar calcAvgAbsDiff(const cv::Mat& src1, const cv::Mat& src2, const cv::Rect& currRect);

// 计算相关系数函数
cv::Scalar calcCenterCorrRatio(const cv::Mat& src1, const cv::Mat& src2, const cv::Rect& rect);
cv::Scalar calcCenterCorrRatio(const cv::Mat& src1, const cv::Mat& src2, const cv::Rect& rect, const cv::Mat& maskImage);
cv::Scalar calcOriginCorrRatio(const cv::Mat& src1, const cv::Mat& src2, const cv::Rect& rect);
cv::Scalar calcOriginCorrRatio(const cv::Mat& src1, const cv::Mat& src2, const cv::Rect& rect, const cv::Mat& maskImage);

// 计算范数
void calcElemWiseL1Norm(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst);
void calcElemWiseL2Norm(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst);

// 计算梯度
void calcThresholdedGradient(const cv::Mat& src, cv::Mat& dst, double thres);
void calcGradient(const cv::Mat& src, cv::Mat& dst, double scale = 1.0);

// 计算绝对值
template<typename SrcType, typename DstType>
void calcAbs(const std::vector<SrcType>& src, std::vector<DstType>& dst)
{
    int length = src.size();
    std::vector<DstType> tempArr(length);
    for (int i = 1; i < length - 1; i++)
    {
        tempArr[i] = fabs(double(src[i]));
    }
    dst.swap(tempArr);
}

// 计算差分
template<typename SrcType, typename DstType>
void calcDerivative(const std::vector<SrcType>& src, std::vector<DstType>& dst)
{
    int length = src.size();
    std::vector<DstType> tempArr(length); 
    tempArr[0] = 0;
    tempArr[length - 1] = 0;
    for (int i = 1; i < length - 1; i++)
    {
        tempArr[i] = double(src[i]) - double(src[i - 1]);
    }
    dst.swap(tempArr);
}

template<typename SrcType, typename DstType>
void calcAbsDerivative(const std::vector<SrcType>& src, std::vector<DstType>& dst)
{
    int length = src.size();
    std::vector<DstType> tempArr(length);
    tempArr[0] = 0;
    tempArr[length - 1] = 0;
    for (int i = 1; i < length - 1; i++)
    {
        tempArr[i] = fabs(double(src[i]) - double(src[i - 1]));
    }
    dst.swap(tempArr);
}

// 计算滑动窗口中的最大值、最小值、中值和均值
template<typename Type>
void localMax(std::vector<Type>& arr, int winLength)
{
    if (winLength <= 2)
        return;
    
    if (winLength % 2 == 0)
        winLength--; 
    int arrLength = arr.size();
    std::vector<Type> tempArr(arrLength);
    int initLeft, initRight, actualLeft, actualRight;
    for (int i = 0; i < arrLength; i++)
    {
        initLeft = i - winLength / 2;
        initRight = i + winLength / 2;
        actualLeft = initLeft < 0 ? 0 : initLeft;
        actualRight = initRight > arrLength - 1 ? arrLength -1 : initRight;
        tempArr[i] = arr[actualLeft];
        for (int j = actualLeft + 1; j <= actualRight; j++)
            tempArr[i] = tempArr[i] > arr[j] ? tempArr[i] : arr[j];
    }
    arr.swap(tempArr);
}

template<typename SrcType, typename DstType>
void localMax(const std::vector<SrcType>& src, std::vector<DstType>& dst, int winLength)
{
    int arrLength = src.size();
    std::vector<DstType> tempArr(arrLength);
    if (winLength <= 2)
    {
        for (int i = 0; i < arrLength; i++)
            tempArr[i] = src[i];
        dst.swap(tempArr);
        return;
    }
    
    if (winLength % 2 == 0)
        winLength--;    
    int initLeft, initRight, actualLeft, actualRight;
    for (int i = 0; i < arrLength; i++)
    {
        initLeft = i - winLength / 2;
        initRight = i + winLength / 2;
        actualLeft = initLeft < 0 ? 0 : initLeft;
        actualRight = initRight > arrLength - 1 ? arrLength -1 : initRight;
        tempArr[i] = src[actualLeft];
        for (int j = actualLeft + 1; j <= actualRight; j++)
            tempArr[i] = tempArr[i] > src[j] ? tempArr[i] : src[j];
    }
    dst.swap(tempArr);
}

template<typename Type>
void localMin(std::vector<Type>& arr, int winLength)
{
    if (winLength <= 2)
        return;
    
    if (winLength % 2 == 0)
        winLength--; 
    int arrLength = arr.size();
    std::vector<Type> tempArr(arrLength);
    int initLeft, initRight, actualLeft, actualRight;
    for (int i = 0; i < arrLength; i++)
    {
        initLeft = i - winLength / 2;
        initRight = i + winLength / 2;
        actualLeft = initLeft < 0 ? 0 : initLeft;
        actualRight = initRight > arrLength - 1 ? arrLength -1 : initRight;
        tempArr[i] = arr[actualLeft];
        for (int j = actualLeft + 1; j <= actualRight; j++)
            tempArr[i] = tempArr[i] < arr[j] ? tempArr[i] : arr[j];
    }
    arr.swap(tempArr);
}

template<typename SrcType, typename DstType>
void localMin(const std::vector<SrcType>& src, std::vector<DstType>& dst, int winLength)
{
    int arrLength = src.size();
    std::vector<DstType> tempArr(arrLength);
    if (winLength <= 2)
    {
        for (int i = 0; i < arrLength; i++)
            tempArr[i] = src[i];
        dst.swap(tempArr);
        return;
    }
    
    if (winLength % 2 == 0)
        winLength--;    
    int initLeft, initRight, actualLeft, actualRight;
    for (int i = 0; i < arrLength; i++)
    {
        initLeft = i - winLength / 2;
        initRight = i + winLength / 2;
        actualLeft = initLeft < 0 ? 0 : initLeft;
        actualRight = initRight > arrLength - 1 ? arrLength -1 : initRight;
        tempArr[i] = src[actualLeft];
        for (int j = actualLeft + 1; j <= actualRight; j++)
            tempArr[i] = tempArr[i] < src[j] ? tempArr[i] : src[j];
    }
    dst.swap(tempArr);
}

template<typename Type>
void localMedian(std::vector<Type>& arr, int winLength)
{
    if (winLength <= 2)
        return;
    
    if (winLength % 2 == 0)
        winLength--; 
    int arrLength = arr.size();
    std::vector<Type> tempArr(arrLength);
    std::vector<Type> opVector(winLength);
    int initLeft, initRight, actualLeft, actualRight;
    for (int i = 0; i < arrLength; i++)
    {
        initLeft = i - winLength / 2;
        initRight = i + winLength / 2;
        actualLeft = initLeft < 0 ? 0 : initLeft;
        actualRight = initRight > arrLength - 1 ? arrLength -1 : initRight;
        opVector.clear();
        for (int j = actualLeft; j <= actualRight; j++)
            opVector.push_back(arr[j]);
        sort(opVector.begin(), opVector.end());
        tempArr[i] = opVector[(actualRight - actualLeft + 1) / 2];
    }
    arr.swap(tempArr);
}

template<typename SrcType, typename DstType>
void localMedian(const std::vector<SrcType>& src, std::vector<DstType>& dst, int winLength)
{
    int arrLength = src.size();
    std::vector<DstType> tempArr(arrLength);
    if (winLength <= 2)
    {
        for (int i = 0; i < arrLength; i++)
            tempArr[i] = src[i];
        dst.swap(tempArr);
        return;
    }
    
    if (winLength % 2 == 0)
        winLength--; 
    std::vector<SrcType> opVector(winLength);
    int initLeft, initRight, actualLeft, actualRight;
    for (int i = 0; i < arrLength; i++)
    {
        initLeft = i - winLength / 2;
        initRight = i + winLength / 2;
        actualLeft = initLeft < 0 ? 0 : initLeft;
        actualRight = initRight > arrLength - 1 ? arrLength -1 : initRight;
        opVector.clear();
        for (int j = actualLeft; j <= actualRight; j++)
            opVector.push_back(src[j]);
        sort(opVector.begin(), opVector.end());
        tempArr[i] = opVector [(actualRight - actualLeft + 1) / 2];
    }
    dst.swap(tempArr);
}

template<typename Type>
void localAbsDiff(std::vector<Type>& arr, int winLength)
{
    if (winLength <= 2)
        return;
    
    if (winLength % 2 == 0)
        winLength--; 
    int arrLength = arr.size();
    std::vector<Type> tempArr(arrLength);
    std::vector<Type> opVector(winLength);
    int initLeft, initRight, actualLeft, actualRight;
    for (int i = 0; i < arrLength; i++)
    {
        initLeft = i - winLength / 2;
        initRight = i + winLength / 2;
        actualLeft = initLeft < 0 ? 0 : initLeft;
        actualRight = initRight > arrLength - 1 ? arrLength -1 : initRight;
        opVector.clear();
        for (int j = actualLeft; j <= actualRight; j++)
            opVector.push_back(arr[j]);
        sort(opVector.begin(), opVector.end());
        tempArr[i] = fabs(double(opVector[actualRight - actualLeft]) - double(opVector[0]));
    }
    arr.swap(tempArr);
}

template<typename SrcType, typename DstType>
void localAbsDiff(const std::vector<SrcType>& src, std::vector<DstType>& dst, int winLength)
{
    int arrLength = src.size();
    std::vector<DstType> tempArr(arrLength);
    if (winLength <= 2)
    {
        for (int i = 0; i < arrLength; i++)
            tempArr[i] = src[i];
        dst.swap(tempArr);
        return;
    }
    
    if (winLength % 2 == 0)
        winLength--; 
    std::vector<SrcType> opVector(winLength);
    int initLeft, initRight, actualLeft, actualRight;
    for (int i = 0; i < arrLength; i++)
    {
        initLeft = i - winLength / 2;
        initRight = i + winLength / 2;
        actualLeft = initLeft < 0 ? 0 : initLeft;
        actualRight = initRight > arrLength - 1 ? arrLength -1 : initRight;
        opVector.clear();
        for (int j = actualLeft; j <= actualRight; j++)
            opVector.push_back(src[j]);
        sort(opVector.begin(), opVector.end());
        tempArr[i] = fabs(double(opVector[actualRight - actualLeft]) - double(opVector[0]));
    }
    dst.swap(tempArr);
}

template<typename Type>
void localMean(std::vector<Type>& arr, int winLength)
{
    if (winLength <= 2)
        return;
    
    if (winLength % 2 == 0)
        winLength--; 
    int arrLength = arr.size();
    std::vector<Type> tempArr(arrLength);
    double tempVal;
    int initLeft, initRight, actualLeft, actualRight;
    for (int i = 0; i < arrLength; i++)
    {
        initLeft = i - winLength / 2;
        initRight = i + winLength / 2;
        actualLeft = initLeft < 0 ? 0 : initLeft;
        actualRight = initRight > arrLength - 1 ? arrLength -1 : initRight;
        tempVal = 0;
        for (int j = actualLeft; j <= actualRight; j++)
            tempVal += arr[j];
        tempArr[i] = tempVal / winLength;
    }
    arr.swap(tempArr);
}

template<typename SrcType, typename DstType>
void localMean(const std::vector<SrcType>& src, std::vector<DstType>& dst, int winLength)
{
    int arrLength = src.size();
    std::vector<DstType> tempArr(arrLength);
    if (winLength <= 2)
    {
        for (int i = 0; i < arrLength; i++)
            tempArr[i] = src[i];
        dst.swap(tempArr);
        return;
    }
    
    if (winLength % 2 == 0)
        winLength--;    
    double tempVal;
    int initLeft, initRight, actualLeft, actualRight;
    for (int i = 0; i < arrLength; i++)
    {
        initLeft = i - winLength / 2;
        initRight = i + winLength / 2;
        actualLeft = initLeft < 0 ? 0 : initLeft;
        actualRight = initRight > arrLength - 1 ? arrLength -1 : initRight;
        tempVal = 0;
        for (int j = actualLeft; j <= actualRight; j++)
            tempVal += src[j];
        tempArr[i] = tempVal / winLength;
    }
    dst.swap(tempArr);
}

template<typename Type>
void localWeightedMean(std::vector<Type>& arr, std::vector<double>& weights)
{
    int winLength = weights.size();
    if (winLength <= 2)
        return;

    if (winLength % 2 == 0)
        winLength--;
    int arrLength = arr.size();
    std::vector<Type> tempArr(arrLength);
    int initLeft, initRight, actualLeft, actualRight;
    double tempVal;
    for (int i = 0; i < arrLength; i++)
    {
        initLeft = i - winLength / 2;
        initRight = i + winLength / 2;
        actualLeft = initLeft < 0 ? 0 : initLeft;
        actualRight = initRight > arrLength - 1 ? arrLength -1 : initRight;
        tempVal = 0;
        for (int j = actualLeft; j <= actualRight; j++)
            tempVal += arr[j] * weights[j - initLeft];
        tempArr[i] = tempVal;
    }
    arr.swap(tempArr);
}

template<typename SrcType, typename DstType>
void localWeightedMean(const std::vector<SrcType>& src, std::vector<DstType>& dst, std::vector<double>& weights)
{
    int arrLength = src.size();
    std::vector<DstType> tempArr(arrLength);
    int winLength = weights.size();
    if (winLength <= 2)
    {
        for (int i = 0; i < arrLength; i++)
            tempArr[i] = src[i];
        dst.swap(tempArr);
        return;
    }

    if (winLength % 2 == 0)
        winLength--;
    int initLeft, initRight, actualLeft, actualRight;
    double tempVal;
    for (int i = 0; i < arrLength; i++)
    {
        initLeft = i - winLength / 2;
        initRight = i + winLength / 2;
        actualLeft = initLeft < 0 ? 0 : initLeft;
        actualRight = initRight > arrLength - 1 ? arrLength -1 : initRight;
        tempVal = 0;
        for (int j = actualLeft; j <= actualRight; j++)
            tempVal += src[j] * weights[j - initLeft];
        tempArr[i] = tempVal;
    }
    dst.swap(tempArr);
}

}