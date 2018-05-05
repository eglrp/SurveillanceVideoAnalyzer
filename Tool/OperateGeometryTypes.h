#pragma once

#include <opencv2/core/core.hpp>

namespace ztool
{

typedef cv::Size_<double> Size2d;

inline Size2d div(const cv::Size& nom, const cv::Size& den)
{
    return Size2d(den.width == 0 ? 0 :double(nom.width) / double(den.width),
                  den.height == 0 ? 0 : double(nom.height) / double(den.height));
}

inline Size2d div(const Size2d& nom, const Size2d& den)
{
    return Size2d(den.width == 0 ? 0 :nom.width / den.width,
                  den.height == 0 ? 0 : nom.height / den.height);
}

inline Size2d recip(const Size2d& scale)
{
    return Size2d(1.0 / scale.width, 1.0 / scale.height);
}

inline cv::Size mul(const cv::Size& orig, double scale)
{
    return cv::Size(orig.width * scale, orig.height * scale);
}

inline cv::Size mul(double scale, const cv::Size& orig)
{
    return mul(orig, scale);
}

inline cv::Size mul(const cv::Size& orig, const Size2d& scale)
{
    return cv::Size(orig.width * scale.width, orig.height * scale.height);
}

inline cv::Size mul(const Size2d& scale, const cv::Size& orig)
{
    return mul(orig, scale);
}

inline cv::Rect mul(const cv::Rect& rect, double scale)
{
    return cv::Rect(rect.x * scale, rect.y * scale, 
                    rect.width * scale, rect.height * scale);
}

inline cv::Rect mul(double scale, const cv::Rect& rect)
{
    return mul(rect, scale);
}

inline cv::Rect mul(const cv::Rect& orig, const Size2d& scale)
{
    return cv::Rect(orig.x * scale.width, 
                    orig.y * scale.height,
                    orig.width * scale.width,
                    orig.height * scale.height);
}

inline cv::Rect mul(const Size2d& scale, const cv::Rect& orig)
{
    return mul(orig, scale);
}

inline cv::Point mul(const cv::Point& pt, double scale)
{
    return cv::Point(pt.x * scale, pt.y * scale);
}

inline cv::Point mul(double scale, const cv::Point& pt)
{
    return mul(pt, scale);
}

inline cv::Point mul(const cv::Point& pt, const Size2d& scale)
{
    return cv::Point(pt.x * scale.width, pt.y * scale.height);
}

inline cv::Point mul(const Size2d& scale, const cv::Point& pt)
{
    return mul(pt, scale);
}

}