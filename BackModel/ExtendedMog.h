#pragma once
#include <vector>
#include <opencv2/core/core.hpp>
#include "ExportControl.h"

namespace zsfo
{

//! 混合高斯模型
class Z_LIB_EXPORT Mog
{
public:
    //! 使用 image 进行初始化
    /*!
        \param[in] image 格式为 CV_8UC1 或者 CV_8UC3, 其他格式图片会抛出 std::exception 类型的异常
     */
	void init(const cv::Mat& image);
    //! 更新模型, 获取前景图和背景图
    /*!
        \param[in] image 需要处理的图片, 尺寸和格式必须和 init 函数中的图片相同, 否则会抛出 std::exception 类型的异常
        \param[out] foreImage 前景图, 尺寸和 image 相同, 格式为 CV_8UC1
        \param[out] backImage 背景图, 尺寸和格式和 image 相同
        \param[in] noUpdate 指定的矩形区域内, 只进行前景提取, 不更新背景模型
     */
    void update(const cv::Mat& image, cv::Mat& foreImage, cv::Mat& backImage, 
		const std::vector<cv::Rect>& noUpdate = std::vector<cv::Rect>());
    //! 更新模型, 获取前景图
    /*!
        \param[in] image 需要处理的图片, 尺寸和格式必须和 init 函数中的图片相同, 否则会抛出 std::exception 类型的异常
        \param[out] foreImage 前景图, 尺寸和 image 相同, 格式为 CV_8UC1
        \param[in] noUpdate 指定的矩形区域内, 只进行前景提取, 不更新背景模型
     */
	void update(const cv::Mat& image, cv::Mat& foreImage, 
		const std::vector<cv::Rect>& noUpdate = std::vector<cv::Rect>());
    //! 更新模型
    /*!
        \param[in] image 需要处理的图片, 尺寸和格式必须和 init 函数中的图片相同, 否则会抛出 std::exception 类型的异常
        \param[in] noUpdate 指定的矩形区域内, 只进行前景提取, 不更新背景模型
     */
	void update(const cv::Mat& image, 
		const std::vector<cv::Rect>& noUpdate = std::vector<cv::Rect>());
    //! 获取背景图, 格式和尺寸同处理图片相同
	void getBackground(cv::Mat& backImage);
    //! 重置模型
    void reset(void);

private:    
    cv::Size frameSize;     ///< 处理图片的尺寸
    cv::Rect frameRect;     ///< 处理图片的矩形
    int frameType;          ///< 处理图片的格式 CV_8UC1 CV_8UC3    
    int frameCount;         ///< 已经处理的帧数, 用于控制模型更新
	cv::Mat bgmodel;        ///< 存储背景模型
	cv::Mat mask;           ///< 确定每个位置的像素是否更新背景模型
};	

}

