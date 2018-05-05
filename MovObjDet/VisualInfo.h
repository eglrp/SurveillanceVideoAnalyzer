#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include "ExportControl.h"

namespace zsfo
{

class Mog;
//! 图像信息
class Z_LIB_EXPORT VisualInfo
{
public:
    //! 初始化, 给图片分配内存, 初始化混合高斯模型
    /*!
        \param[in] image 用于初始化的图片
     */
    void init(const cv::Mat& image);
    //! 更新函数
    /*!
        用混合高斯模型检测前景, 根据 fullUpdate 参数的值决定是否更新背景模型, 
        计算梯度差值图, 加到混合高斯模型检测出的前景中
        \param[in] image 图片
        \param[out] foreImage 前景图, 已经把梯度差值图加上
        \param[out] backImage 混合高斯模型的背景图
        \param[out] gradDiffImage 梯度差值图
        \param[in] fullUpdate 是否完全更新, 如果是, 检测前景的同时更新背景模型, 否则仅检测前景
        \param[in] rectsNoUpdate 如果 fullUpdate == true, 矩形区域内只检测前景, 不更新混合高斯模型
                                 如果 fullUpdate == false, 本参数不起作用
     */
    void update(const cv::Mat& image, bool fullUpdate = true, 
        const std::vector<cv::Rect>& rectsNoUpdate = std::vector<cv::Rect>());
    void update(const cv::Mat& image, cv::Mat& foreImage, bool fullUpdate = true, 
        const std::vector<cv::Rect>& rectsNoUpdate = std::vector<cv::Rect>());
    void update(const cv::Mat& image, cv::Mat& foreImage, cv::Mat& backImage, bool fullUpdate = true, 
        const std::vector<cv::Rect>& rectsNoUpdate = std::vector<cv::Rect>());
    void update(const cv::Mat& image, cv::Mat& foreImage, cv::Mat& backImage, cv::Mat& gradDiffImage, 
        bool fullUpdate = true, const std::vector<cv::Rect>& rectsNoUpdate = std::vector<cv::Rect>());

private:
    cv::Ptr<Mog> backModel;        ///< 混合高斯模型进行背景建模
    cv::Mat normGrayImage;         ///< 输入图片 image 对应的灰度图
    cv::Mat backGrayImage;         ///< backImage 对应的灰度图
    cv::Mat normGradImage;         ///< normGrayImage 的梯度图
    cv::Mat backGradImage;         ///< backGrayImage 的梯度图

    int width;                     ///< 处理图片的宽度
    int height;                    ///< 处理图片的高度
};

}