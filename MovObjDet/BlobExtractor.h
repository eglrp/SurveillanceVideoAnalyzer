#pragma once
#include <vector>
#include <opencv2/core/core.hpp>
#include "ExportControl.h"

namespace zsfo
{

//! 根据前景图, 彩色图等确定当前帧中前景矩形的类
class Z_LIB_EXPORT BlobExtractor
{
public:
    //! 初始化
    /*!
        \param[in] imageSize 图片的尺寸
        \param[in] path 配置文件的路径, 如果等于 0 则采用默认参数
        \param[in] label 配置文件中参数的标签
     */
	void init(const cv::Size& imageSize, const std::string& path = std::string(), const std::string& label = std::string());
    // 修改配置参数
    /*!
        本函数用于修改控制私有成员函数行为的配置参数
        除了 charRegionRects 之外, 其余参数都是指针形式, 如果不需要对某项参数进行设置, 直接传入 0 值即可
        本函数中的参数的设置如何起作用, 参见 proc 函数的说明

        \param[in] minObjectArea 前景轮廓所包围的面积大于这个值才会被保留, 进行后续处理
        \param[in] minObjectWidth 被允许的前景外接矩形的宽度的最小值
        \param[in] minObjectHeight 被允许的前景外接矩形的高度的最小值
        \param[in] corrRatioCheck 是否进行相关系数检测, 判断前景区域是否和背景区域相似, 
                                  如果检测, 和背景图相关系数较大的前景矩形被删除

        \param[in] charRegionCheck 是否检测前景矩形是否落在字符区域中, 如果检测, 落在字符区域中的前景矩形会被删除
        \param[in] charRegionRects 框定字符区域的所有矩形组成的向量

        \param[in] merge 是否进行合并矩形的操作, 本变量的设置只负责能否进入到进行合并矩形的代码段, 如何进行合并, 还需要看后面三个参数的设置
        \param[in] mergeHori 是否进行合并水平方向上占据的位置相近的上下两个矩形的操作
        \param[in] mergeVert 是否进行合并竖直方向上占据的位置相近的左右两个矩形的操作
        \param[in] mergeBigSmall 是否对有面积重叠的大小差异较大的矩形进行合并的操作

        \param[in] refine 是否进行阴影消除操作, 本变量的设置只负责能否进入到
        \param[in] refineByShape 是否根据前景的轮廓进行阴影消除
        \param[in] refineByGrad 是否根据梯度差值图进行阴影消除
        \param[in] refineByColor 在根据梯度差值图进行阴影消除的函数中, 是否进一步根据颜色信息确定区域是否为阴影
     */
    void setConfigParams(
        const double* minObjectArea = 0, const double* minObjectWidth = 0, const double* minObjectHeight = 0, const bool* corrRatioCheck = 0, 
        const bool* charRegionCheck = 0, const std::vector<cv::Rect>& charRegionRects = std::vector<cv::Rect>(),
        const bool* merge = 0, const bool* mergeHori = 0, const bool* mergeVert = 0, const bool* mergeBigSmall = 0,
        const bool* refine = 0, const bool* refineByShape = 0, const bool* refineByGrad = 0, const bool* refineByColor = 0);
	//! 简单版本的处理函数
	/*!
        根据前景图 foreImage(CV_8UC1) 找前景矩形, 过滤掉较小的矩形, 
        如果设置了相关系数检测, 并且 image 和 backImage 都不为空, 则执行相关系数检测, 
        如果 image 和 backImage 在矩形区域部分相关系数较大, 该矩形也会被滤除
        得到 origRects
        用 origRects 合并矩形, 得到 rects
        操作顺序是合并竖直方向上位置相近的矩形, 合并水平方向上位置相近的矩形, 合并大小差异较大的矩形
        从 rects 中判定是否有在同一区域稳定了一段时间的矩形, 被判定为稳定的矩形传给 stableRects
		使用该函数时, 有关阴影消除的参数都不起作用
     */
	void proc(cv::Mat& foreImage, const cv::Mat& image, const cv::Mat& backImage, 
        std::vector<cv::Rect>& rects, std::vector<cv::Rect>& stableRects);
    //! 处理函数
    /*!
        根据彩色图 image(CV_8UC3), 前景图 foreImage(CV_8UC1), 背景图 backImage(CV_8UC3), 
        彩色图和背景图的梯度差值图 gradDiffImage(CV_8UC1) 找前景矩形

        1 找矩形
        根据前景图找矩形, 过滤掉较小的矩形, 
        如果设置了相关系数检测, 并且 image 和 backImage 都不为空, 则执行相关系数检测, 
        如果 image 和 backImage 在矩形区域部分相关系数较大, 该矩形也会被滤除
        得到 origRects        
        2 用 origRects 合并矩形, 得到 mergedRects
        操作顺序是合并竖直方向上位置相近的矩形, 合并水平方向上位置相近的矩形, 合并大小差异较大的矩形
        3 对 mergedRects 进行阴影消除, 得到 refinedRects
        操作顺序是 
        先根据 foreImage 按形状进行阴影消除, 
        再根据 gradDiffImage 按梯度值进行阴影消除, 期间会根据配置参数和 normalImage 和 backImage 是否存在根据颜色确定区域是否为阴影
        refinedRects 传出为 rects
        4 从 origRects 矩形中判定是否有在同一区域稳定了一段时间的矩形, 被判定为稳定的矩形传给 stableRects
     */
	void proc(cv::Mat& foreImage, const cv::Mat& image, const cv::Mat& backImage, const cv::Mat& gradDiffImage, 
        std::vector<cv::Rect>& rects, std::vector<cv::Rect>& stableRects);
    //! 在 image 上用颜色 color 画字符区域
	void drawCharRect(cv::Mat& image, const cv::Scalar& color);
	//! 在 image 上用颜色 color 画稳定的矩形区域
	void drawStableRects(cv::Mat& image, const cv::Scalar& color);
    //! 在 image 上用颜色 color 画最终矩形区域
    void drawFinalRects(cv::Mat& image, const cv::Scalar& color);

private:
	class Impl;
	cv::Ptr<Impl> ptrImpl;
};

}// end zsfo