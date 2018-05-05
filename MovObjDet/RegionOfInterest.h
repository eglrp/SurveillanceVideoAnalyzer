#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/core/core.hpp>
#include "ExportControl.h"

namespace zsfo
{

//! 感兴趣区域设置
struct Z_LIB_EXPORT RegionOfInterest
{
    //! 默认构造函数
    RegionOfInterest(void) {};
    //! 拷贝构造函数, 深拷贝 mask 成员变量
    RegionOfInterest(const RegionOfInterest& roi);
    //! 等号重载, 深拷贝 mask 成员变量
    RegionOfInterest& operator=(const RegionOfInterest& roi);
    //! 通过点向量进行初始化
    /*!
        \param[in] label 本感兴趣区域的标签, 打印参数时需要
        \param[in] imageSize 画面的尺寸
        \param[in] defIncludedRegion 表示用后面的参数 imagePoints 包围的区域是感兴趣区域还是不感兴趣区域
        \param[in] imagePoints 多边形的顶点, 用来表示感兴趣区域或者不感兴趣区域, 也可以是线段, 但此时 imagePoints.size() 必须等于 1
     */
    void init(const std::string& label, const cv::Size& imageSize, bool defIncludedRegion = true,
              const std::vector<std::vector<cv::Point> >& imagePoints = std::vector<std::vector<cv::Point> >());
    //! 通过配置文件初始化
    /*!
        \param[in] imageSize 画面的尺寸
        \param[in] path 配置文件路径
        \param[in] label 配置文件的标签
        配置文件格式如下
        [RegionOfInterest]
        #define_included_region 0
        #num_of_roi  3
        #roi 100 100 50 200 175 175 
        #roi 150 50 150 150 300 150 300 10
        #roi 200 50 200 200 300 100 300 10 200 0
        其中 [RegionOfInterest] 是标签, 标签必须独占一行, 可以使用任意名称
        接下来必须是 #define_included_region 空格后跟值 0 或者 1, 独占一行, 不能分行
        接下来必须是 #num_of_roi 空格后跟非负数值, 独占一行, 不能分行
        接下来是一系列 #roi 每行用 横坐标 纵坐标 横坐标 纵坐标 ... 的方式表示多边形区域的顶点
        每个多边形区域的点数必须大于 2, 即数字的数量必须大于 4
     */
    void init(const cv::Size& imageSize, const std::string& path, const std::string& label);
    //! 过滤掉与感兴趣区域不相交的矩形
    /*!
        \param[in, out] rects 需要处理的矩形和处理结果
     */
    void filterNotIntersecting(std::vector<cv::Rect>& rects) const;
    //! 过滤掉与感兴趣区域不相交的矩形
    /*!
        \param[in, out] rects 需要处理的矩形和处理结果
        \param[in] ratio 落在感兴趣区域的面积占整个矩形的面积的比例小于这个值, 则过滤这个矩形
     */
    void filterNotIntersecting(std::vector<cv::Rect>& rects, const double ratio) const;
    //! 对 testRect 网格化, 检测网格化的点是否至少有一个点落在感兴趣区域内
    /*!
        \param[in, out] rects 需要处理的矩形和处理结果
     */
    bool intersects(const cv::Rect& testRect) const;
    //! 对 testRect 网格化, 检测网格化的点是落在感兴趣区域中的数量占所有点的数量的比例是否大于阈值
    /*!
        \param[in, out] rects 需要处理的矩形和处理结果
        \param[in] ragio 阈值
     */
    bool intersects(const cv::Rect& testRect, const double ratio) const;
    //! 感兴趣区域是否包含 testPoint
    /*!
        \param[in, out] rects 需要处理的矩形和处理结果
     */
    bool contains(const cv::Point& testPoint) const;
    //! 画出感兴趣区域
    /*!
        \param[in] image 在这张图上话感兴趣区域
        \param[in] color 用这个颜色话感兴趣区域的边缘
     */
    void draw(cv::Mat& image, const cv::Scalar& color) const;

    bool isFullSize;       ///< 感兴趣区域是否几乎布满整个画面
    int width;             ///< 画面的宽度
    int height;            ///< 画面的高度
    cv::Rect imageRect;    ///< 画面对应的矩形
    cv::Mat mask;          ///< 感兴趣区域图, 类型为 CV_8UC1, 感兴趣区域内值等于 255, 其他区域值等于 0
    bool defIncRegion;     ///< 多边形的顶点包围的是否为感兴趣区域
    std::vector<std::vector<cv::Point> > vertices;   ///< 包围感兴趣区域或者不感兴趣区域的多边形的顶点
};

//! 圆形区域
struct Circle
{
    //! 默认构造函数
    Circle(void) : center(0, 0), radius(0), sqrRadius(0) {};
    //! 根据圆心 center 和半径 radius 构造圆
    Circle(const cv::Point& center_, int radius_)
        : center(center_.x, center_.y), radius(radius_), sqrRadius(radius_ * radius_) {};
    //! 根据直径上的两个端点 pt1 和 pt2 构造圆
    Circle(const cv::Point& pt1, const cv::Point& pt2)
    {
        cv::Point2d sum = pt1 + pt2;
        center.x = sum.x / 2;
        center.y = sum.y / 2;
        cv::Point2d diff = pt1 - pt2;
        sqrRadius = diff.dot(diff) * 0.25;
        radius = sqrt(sqrRadius);
    }
    //! 判断点 pt 是否落在圆内
    bool contains(const cv::Point& pt) const
    {
        cv::Point2d diff = cv::Point2d(pt.x, pt.y) - center; 
        return diff.dot(diff) < sqrRadius;
    }
    cv::Point2d center;  ///< 圆心
    double radius;       ///< 半径
    double sqrRadius;    ///< 半径的平方
};

//! 两条平行线中间的区域
struct ParallelLineInterior
{
    //! 默认构造函数
    ParallelLineInterior(void)
        : a1(0), b1(0), c1(0), a2(0), b2(0), c2(0), sign1(0), sign2(0) {}
    //! 用两个点 pt1 和 pt2 构造, 两个点分别位于两条直线上, 并且 pt1 和 pt2 之间的距离等于两条平行钱间的距离
    ParallelLineInterior(const cv::Point& pt1, const cv::Point& pt2)
        : a1(0), b1(0), c1(0), a2(0), b2(0), c2(0), sign1(0), sign2(0) 
    {
        if (pt1 == pt2) return; 

        if (pt1.x == pt2.x)
        {
            a1 = 0; b1 = 1; c1 = -pt1.y;
            a2 = 0; b2 = 1; c2 = -pt2.y;
        }
        else if (pt1.y == pt2.y)
        {
            a1 = 1; b1 = 0; c1 = -pt1.x;
            a2 = 1; b2 = 0; c2 = -pt2.x;
        }
        else
        {
            double kk, ss, bb;
            kk = double(pt1.y - pt2.y) / double(pt1.x - pt2.x);
            kk = -1.0 / kk;
            ss = sqrt(kk * kk + 1.0);

            a1 = a2 = kk / ss;
            b1 = b2 = -1.0 / ss;

            bb = pt1.y - kk * pt1.x;           
            c1 = bb / ss;

            bb = pt2.y - kk * pt2.x;
            c2 = bb / ss;
        }

        cv::Point center;
        center.x = (pt1.x + pt2.x) / 2;
        center.y = (pt1.y + pt2.y) / 2;
        sign1 = (center.x * a1 + center.y * b1 + c1) > 0 ? 1 : -1;
        sign2 = (center.x * a2 + center.y * b2 + c2) > 0 ? 1 : -1;
    }
    //! 判断点 pt 是否在两条平行线内部
    bool contains(const cv::Point& pt) const
    {
        if (sign1 == 0 && sign2 == 0) return false;
        
        double val1 = pt.x * a1 + pt.y * b1 + c1;
        double val2 = pt.x * a2 + pt.y * b2 + c2;
        if (abs(val1) < 0.001)
            return val2 * sign2 > 0;
        if (abs(val2) < 0.001)
            return val1 * sign1 > 0;
        return (val1 * sign1 > 0) && (val2 * sign2 > 0);
    }

    // a1 * x + b1 * y + c1 = 0, a1 ^ 2 + b1 ^ 2 = 1
    // a2 * x + b2 * y + c2 = 0, a2 ^ 2 + b2 ^ 2 = 1
    double a1, b1, c1;
    double a2, b2, c2;
    int sign1, sign2;
};

//! 线段
struct Z_LIB_EXPORT LineSegment
{
    //! 根据配置文件进行初始化
    /*
        \param[in] path 配置文件的路径
        \param[in] label 配置文件中标识线段参数的标签
     */
    void init(const std::string& path, const std::string& label);
    //! 根据线段的两个端点 begPoint 和 endPoint 初始化线段, begSidePoint 表示线段所在直线指定的一侧
    void init(const cv::Point& begPoint, const cv::Point& endPoint, 
        const cv::Point& begSidePoint = cv::Point(-1, -1));
    //! 计算点 pt 到线段的距离
    /*!
        如果 pt 在成员变量 region 的区域内部, 则计算 pt 到线段所在直线的距离
        否则计算 pt 到线段两个端点的距离, 然后取最小值
     */
    double distTo(const cv::Point& pt) const;
    //! 判断点 pt 是否靠近线段, pt 到线段的距离小于 dist 则返回 true, 否则返回 false
    bool closeTo(const cv::Point& pt, int dist = 5) const;
    //! 判断点 pt 是否在初始化中指定的一侧
    bool inBegSide(const cv::Point& pt) const;
    //! 在 image 中用颜色 color 画出线段 
    void drawLineSegment(cv::Mat& image, const cv::Scalar& color) const;

    cv::Point beg, end;           ///< 线段的两个端点
    ParallelLineInterior region;  ///< 计算点到线段的距离中需要指定的区域
    //Circle region;
    //cv::Rect region;
    double a, b, c;           // ax + by + c = 0 and a ^ 2 + b ^ 2 = 1
    int signOnBegSide;        ///< 指定一侧的符号
};

//! 虚拟线圈
struct Z_LIB_EXPORT VirtualLoop
{
    //! 根据配置文件对虚拟线圈进行初始化
    /*!
        \param[in] path 配置文件的路径
        \param[in] label 配置文件中标记初始化参数的标签
     */
    void init(const std::string& path, const std::string& label);
    //! 初始化
    /*!
        \param[in] label 本虚拟线圈的标签, 打印参数时需要
        \param[in] points 图像坐标
     */
    void init(const std::string& label, const std::vector<cv::Point>& points);
    //! 判断检测矩形是否与虚拟线圈有交集
    bool intersects(const cv::Rect& testRect) const;
    //! 判断检测点是否落在虚拟线圈中
    bool contains(const cv::Point& testPoint) const;
    //! 判断检测点是否在线圈底部边界以下
    bool overTopBound(const cv::Point& testPoint) const;
    //! 判断检测点是否在线圈上部边界以上
    bool belowBottomBound(const cv::Point& testPoint) const;
    //! 判断监测点是否在线圈左边界的左边
    bool leftToLeftBound(const cv::Point& testPoint) const;
    //! 判断监测点是否在线圈右边界的右边
    bool rightToRightBound(const cv::Point& testPoint) const;
    //! 判断检测点是靠近上边界还是下边界
    bool closerToTopThanToBottom(const cv::Point& testPoint) const;
    //! 判断检测点是靠近左边界还是右边界
    bool closerToLeftThanToRight(const cv::Point& testPoint) const;
    //! 计算某个点到上下左右边界的图像距离
    double distToLeftBound(const cv::Point& point) const;
    double distToRightBound(const cv::Point& point) const;
    double distToTopBound(const cv::Point& point) const;
    double distToBottomBound(const cv::Point& point) const;
    //! 画出虚拟线圈
    void drawLoop(cv::Mat& image, const cv::Scalar& color) const;
        
    /*! 
        图像中虚拟线圈四条直线的方程
        对于左边界和右边界的直线，方程采用 x = k * y + b 的形式
        对于上边界和下边界的直线，方程采用 y = k * x + b 的形式
    */
    double leftK, leftB;
    double rightK, rightB;
    double topK, topB;
    double bottomK, bottomB;

    //! 图像坐标中虚拟线圈的四个顶点，单位是像素
    cv::Point vertices[4];
};

inline bool RegionOfInterest::contains(const cv::Point& testPoint) const
{
    if (!imageRect.contains(testPoint))
        return false;
    return mask.at<unsigned char>(testPoint) > 0;
};

inline bool VirtualLoop::contains(const cv::Point& testPoint) const
{
    return (testPoint.x >= leftK * testPoint.y + leftB) &&
        (testPoint.x <= rightK * testPoint.y + rightB) &&
        (testPoint.y >= topK * testPoint.x + topB) &&
        (testPoint.y <= bottomK * testPoint.x + bottomB);
}

inline bool VirtualLoop::belowBottomBound(const cv::Point& testPoint) const
{
    return testPoint.y > bottomK * testPoint.x + bottomB;
}

inline bool VirtualLoop::overTopBound(const cv::Point& testPoint) const
{
    return testPoint.y < topK * testPoint.x + topB;
}

inline bool VirtualLoop::leftToLeftBound(const cv::Point& testPoint) const
{
    return testPoint.x < leftK * testPoint.y + leftB;
}

inline bool VirtualLoop::rightToRightBound(const cv::Point& testPoint) const
{
    return testPoint.x > rightK * testPoint.y + rightB;
}

inline bool VirtualLoop::closerToTopThanToBottom(const cv::Point& testPoint) const
{
    return (fabs(topK * testPoint.x - testPoint.y + topB) <
            fabs(bottomK * testPoint.x - testPoint.y + bottomB));
}

inline bool VirtualLoop::closerToLeftThanToRight(const cv::Point& testPoint) const
{
    return (fabs(leftK * testPoint.y - testPoint.x + leftB) <
            fabs(rightK * testPoint.y - testPoint.x + rightB));
}

inline double VirtualLoop::distToLeftBound(const cv::Point& point) const
{
    return fabs(leftK * point.y + leftB - point.x) / sqrt(1 + leftK * leftK);
}

inline double VirtualLoop::distToRightBound(const cv::Point& point) const
{
    return fabs(rightK * point.y + rightB - point.x) / sqrt(1 + rightK * rightK);
}

inline double VirtualLoop::distToTopBound(const cv::Point& point) const
{
    return fabs(topK * point.x + topB - point.y) / sqrt(1 + topK * topK);
}

inline double VirtualLoop::distToBottomBound(const cv::Point& point) const
{
    return fabs(bottomK * point.x + bottomB - point.y) / sqrt(1 + bottomK * bottomK);
}

inline void VirtualLoop::drawLoop(cv::Mat& image, const cv::Scalar& color) const
{
    cv::line(image, vertices[0], vertices[1], color, 2);
    cv::line(image, vertices[1], vertices[2], color, 2);
    cv::line(image, vertices[2], vertices[3], color, 2);
    cv::line(image, vertices[3], vertices[0], color, 2);
}

inline int pointDist(const cv::Point& pt1, const cv::Point& pt2)
{
    cv::Point diff = pt1 - pt2;
    return std::sqrt(double(diff.dot(diff)));
}

inline double LineSegment::distTo(const cv::Point& pt) const
{
    if (region.contains(pt))
        return abs(a * pt.x + b * pt.y + c);

    return std::min(pointDist(pt, beg), pointDist(pt, end)); 
}

inline bool LineSegment::closeTo(const cv::Point& pt, int dist) const
{
    if (region.contains(pt))
        return abs(a * pt.x + b * pt.y + c) < dist;

    return pointDist(pt, beg) < dist || pointDist(pt, end) < dist; 
}

inline bool LineSegment::inBegSide(const cv::Point& pt) const
{
    return (a * pt.x + a * pt.y + c) * signOnBegSide > 0;
}

inline void LineSegment::drawLineSegment(cv::Mat& image, const cv::Scalar& color) const
{
    cv::line(image, beg, end, color, 2);
}

}