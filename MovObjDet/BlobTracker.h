#pragma once

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include "ExportControl.h"

namespace zsfo
{

struct RegionOfInterest;
struct LineSegment;
struct VirtualLoop;

//! 记录快照模式
struct RecordMode
{
    enum 
    {
        CrossTriBoundVisualRecord = 0,     ///< 给定抓拍线圈, 运动目标跨越左右下边界时抓拍
        CrossBottomBoundVisualRecord = 1,  ///< 给定抓拍线圈, 运动目标跨越下边界时抓拍
        CrossLineSegmentVisualRecord = 2,  ///< 给定线段, 运动目标跨越线段时抓拍
        MultiVisualRecord = 3,             ///< 存多幅快照
        NoVisualRecord = 4                 ///< 只记录矩形框历史
    };
};

//! 保存快照图片类型
struct SaveImageMode
{
    enum
    {
        SaveScene = 1,         ///< 保存全景图
        SaveSlice = 2,         ///< 保存目标截图
        SaveMask = 4           ///< 保存目标前景图
    };
};

//! 输出运动目标的单帧历史记录
struct ObjectRecord
{
	//! 构造函数
    ObjectRecord(void) : time(0), number(0)	{};

	long long int time;        ///< 时间戳
	int number;                ///< 帧编号
	cv::Rect normRect;         ///< 归一化帧中的矩形
	cv::Rect origRect;         ///< 原始尺寸帧中的矩形
    cv::Mat image;             ///< 目标截图, 从原始帧中的 origRect 中截取
};

//! 输出运动目标的快照记录
struct ObjectVisualRecord
{
    //! 构造函数
    ObjectVisualRecord(void) : time(0), number(0), bound(-1), cross(-1), direction(-1) {};

    long long int time;        ///< 快照图片的时间戳
	int number;                ///< 快照图片的帧编号
	cv::Mat scene;             ///< 快照图片对应的全景图, 视频帧的尺寸
	cv::Mat mask;              ///< 快照图片对应的前景图, 视频帧的尺寸
    cv::Mat slice;             ///< 快照图片
	cv::Rect rect;             ///< 运动目标在全景图上的矩形
	int bound;                 ///< 快照时跨越线圈的边界 1 左边界 2 右边界 3 下边界 -1 未知
	int cross;                 ///< 快照时是否进线圈 1 是 0 不是 -1 未知
	int direction;             ///< 快照时的行驶方向 1 从左到右 2 从右到左 3 从上到下 4 从下到上 -1 未知
};

//! 输出的运动目标结构体
struct ObjectInfo
{
	//! 构造函数
    ObjectInfo(void)
        : ID(0), isFinal(0), hasHistory(0), hasVisualHistory(0), speed(0), velocity(0)
	{};

	// 每帧处理结束都会返回的信息
	int ID;                    ///< 运动目标编号
	cv::Rect currRect;         ///< 运动目标在当前帧中的矩形
	int isFinal;               ///< 是否跟踪结束	

	// 以下信息 isFinal = 1 时才能获取

	// 历史信息
	int hasHistory;            ///< 是否有历史轨迹信息
	std::vector<ObjectRecord> history;     ///< 历史轨迹

    // 快照信息
    int hasVisualHistory;      ///< 是否有快照图片
    std::vector<ObjectVisualRecord> visualHistory; ///< 快照图片
	
	double speed;              ///< 速度，标清版中单位为像素/秒
	double velocity;           ///< 速度，高清版中单位为千米/时
};

//! 输出的静态目标结构体
struct StaticObjectInfo
{
	//! 构造函数
    StaticObjectInfo(void) : ID(0) {};

	int ID;          ///< 目标编号
	cv::Rect rect;   ///< 目标的位置
};

//! 表示原始尺寸和归一化尺寸的结构体
struct SizeInfo
{
	//! 初始化成员变量
    /*!
        \param[in] origSize 原始尺寸
        \param[in] normSize 归一化尺寸
     */
    void create(const cv::Size& origSize, const cv::Size& normSize);
	
	int normWidth;    ///< 归一化宽度
	int normHeight;   ///< 归一化高度
	int origWidth;    ///< 原始宽度
	int origHeight;   ///< 原始高度
	double horiScale; ///< 水平尺度因子
	double vertScale; ///< 竖直尺度因子
};

inline void SizeInfo::create(const cv::Size& origSize, const cv::Size& normSize)
{
	normWidth = normSize.width;
	normHeight = normSize.height;
	origWidth = origSize.width;
	origHeight = origSize.height;
	horiScale = double(origWidth) / double(normWidth);
	vertScale = double(origHeight) / double(normHeight);
}

//! 管理所有跟踪对象和所有共享资源的类
class Z_LIB_EXPORT BlobTracker
{
public:
    //! 构造函数
    BlobTracker(void) {};
    //! 无抓拍图片初始化
    /*!
        \param[in] observedRegion 观测和跟踪区域
        \param[in] sizesOrigAndNorm 原始尺寸和归一化尺寸
        \param[in] path 配置文件路径
     */
    void init(const RegionOfInterest& observedRegion, const SizeInfo& sizesOrigAndNorm, 
        const std::string& path = std::string());
	//! 按跨线抓拍的方式初始化
    /*!
        \param[in] observedRegion 观测和跟踪区域
        \param[in] crossLine 抓拍图片使用的线段, 当矩形中心靠近线圈时才进行抓拍, 线段应当位于观测和跟踪区域内
        \param[in] sizesOrigAndNorm 原始尺寸和归一化尺寸
        \param[in] saveImageMode 存图模式, 选择是否保存全景图, 目标截图和目标前景图
        \param[in] path 配置文件路径
     */
    void initLineSegment(const RegionOfInterest& observedRegion, const LineSegment& crossLine, 
        const SizeInfo& sizesOrigAndNorm, int saveImageMode, const std::string& path = std::string());
    //! 按跨线圈底部边界抓拍的方式初始化
    /*!
        \param[in] observedRegion 观测和跟踪区域
        \param[in] catchLoop 抓拍图片使用的线圈, 本线圈应当位于观测和跟踪区域内部
        \param[in] sizesOrigAndNorm 原始尺寸和归一化尺寸
        \param[in] saveImageMode 存图模式, 选择是否保存全景图, 目标截图和目标前景图
        \param[in] path 配置文件路径
     */
	void initBottomBound(const RegionOfInterest& observedRegion, const VirtualLoop& catchLoop, 
        const SizeInfo& sizesOrigAndNorm, int saveImageMode, const std::string& path = std::string());
    //! 按跨越线圈左右下三边界抓拍的方式初始化
    /*!
        \param[in] observedRegion 观测和跟踪区域
        \param[in] catchLoop 抓拍图片使用的线圈, 本线圈应当位于观测和跟踪区域内部
        \param[in] sizesOrigAndNorm 原始尺寸和归一化尺寸
        \param[in] saveImageMode 存图模式, 选择是否保存全景图, 目标截图和目标前景图
        \param[in] path 配置文件路径
     */
    void initTriBound(const RegionOfInterest& observedRegion, const VirtualLoop& catchLoop, 
        const SizeInfo& sizesOrigAndNorm, int saveImageMode, const std::string& path = std::string());
    //! 按保存多幅历史图片的方式初始化
    /*!
        \param[in] observedRegion 观测和跟踪区域
        \param[in] sizesOrigAndNorm 原始尺寸和归一化尺寸
        \param[in] saveMode 存图模式, 选择是否保存全景图, 目标截图和目标前景图
        \param[in] saveInterval 保存图片的帧间隔
        \param[in] numOfSaved 最多保存多少张图片
        \param[in] path 配置文件路径
     */
    void initMultiRecord(const RegionOfInterest& observedRegion, const SizeInfo& sizesOrigAndNorm, 
        int saveImageMode, int saveInterval, int numOfSaved, const std::string& path = std::string());
    //! 修改一些配置参数
    /*!
        所有函数的传入参数均为指针形式, 只要指针不为空指针, 就会将类的配置参数按给定的值重置
        \param[in] checkTurnAround 是否检测运动目标折返, 如果检测, 并且存在折返, 则删除当前跟踪目标, 建立新的跟踪目标
        \param[in] maxDistRectAndBlob 如果当前帧某个矩形的中心和某个被跟踪对象在上一帧的矩形的中心距离小于这个值, 则满足匹配条件之一
        \param[in] minRatioIntersectToSelf 如果当前帧某个矩形和某个被跟踪对象在上一帧的矩形的交集的面积和当前帧这个矩形的面积的比值大于这个值, 则满足匹配条件之一
        \param[in] minRatioIntersectToBlob 如果当前帧某个矩形和某个被跟踪对象在上一帧的矩形的交集的面积和这个被跟踪对象矩形的面积的比值大于这个值, 则满足匹配条件之一
     */
    void setConfigParams(const bool* checkTurnAround = 0, const double* maxDistRectAndBlob = 0,
        const double* minRatioIntersectToSelf = 0, const double* minRatioIntersectToBlob = 0);
	//! 处理函数
    /*!
	    如果 BlobTracker 的实例按照含有保存历史图片的方式进行初始化, 并且调用这个版本的处理函数, 将不会得到图片
        \param[in] time 时间戳
        \param[in] count 帧编号
        \param[in] rects 当前帧中检测到的矩形
        \param[out] objects 输出信息, 当前处于跟踪状态的目标输出位置等信息, 结束跟踪的目标输出历史轨迹
     */
    void proc(long long int time, int count, const std::vector<cv::Rect>& rects, std::vector<ObjectInfo>& objects);
	//! 处理函数
    /*!
	    如果 BlobTracker 的实例按照无抓拍图片的方式进行初始化, 并且调用这个版本的处理函数, 将不会得到图片
        \param[in] origFrame 原始尺寸的视频帧
        \param[in] foreImage 归一化尺寸的前景图
        \param[in] time 时间戳
        \param[in] count 帧编号
        \param[in] rects 当前帧中检测到的矩形
        \param[out] objects 输出信息, 当前处于跟踪状态的目标输出位置等信息, 结束跟踪的目标输出历史轨迹和快照截图
     */
    void proc(const cv::Mat& origFrame, const cv::Mat& foreImage, 
		long long int time, int count, const std::vector<cv::Rect>& rects, std::vector<ObjectInfo>& objects);
    //! 处理函数
    /*!
	    如果 BlobTracker 的实例按照无抓拍图片的方式进行初始化, 并且调用这个版本的处理函数, 将不会得到图片
        \param[in] origFrame 原始尺寸的视频帧
        \param[in] foreImage 归一化尺寸的前景图
        \param[in] gradDiffImage 归一化尺寸输入图和背景图的梯度图的差值图
        \param[in] lastGradDiffImage 上一次处理时得到的归一化尺寸输入图和背景图的梯度图的差值图
        \param[in] time 时间戳
        \param[in] count 帧编号
        \param[in] rects 当前帧中检测到的矩形
        \param[out] objects 输出信息, 当前处于跟踪状态的目标输出位置等信息, 结束跟踪的目标输出历史轨迹和快照截图
     */
	void proc(const cv::Mat& origFrame, const cv::Mat& foreImage, 
        const cv::Mat& gradDiffImage, const cv::Mat& lastGradDiffImage, 
        long long int time, int count, const std::vector<cv::Rect>& rects, std::vector<ObjectInfo>& objects);
    //! 画跟踪状态
    /*!
        \param[out] frame 归一化尺寸的当前帧
        \param[in] observedRegionColor 观测和跟踪区域的颜色
        \param[in] crossLoopOrLineColor 抓拍图片使用的线圈或者线段的颜色
        \param[in] blobRectColor 运动目标矩形框的颜色
        \param[in] blobHistoryColor 运动目标矩形中心历史轨迹线的颜色
     */
    void drawTrackingState(cv::Mat& frame, const cv::Scalar& observedRegionColor, const cv::Scalar& crossLoopOrLineColor,
        const cv::Scalar& blobRectColor, const cv::Scalar& blobHistoryColor) const;	
    //! 视频处理完最后一帧调用的终止函数
    /*!
        视频已经处理完, 不管是否跟踪结束, 都将运动目标的历史轨迹和抓拍图片输出
     */
    void final(std::vector<ObjectInfo>& objects) const;

private:
    //! 不能拷贝构造和赋值
    BlobTracker(const BlobTracker&);
    BlobTracker& operator=(const BlobTracker&);

	class BlobTrackerImpl;
	cv::Ptr<BlobTrackerImpl> ptrImpl;
};

//! 静态目标跟踪类
class Z_LIB_EXPORT StaticBlobTracker
{
public:
	//! 构造函数
	StaticBlobTracker(void) {};
	//! 析构函数
	~StaticBlobTracker(void) {};

	//! 初始化
	/*!
	    \param[in] observedRegion 观测静态目标的区域
		\param[in] sizesOrigAndNorm 原始尺寸和归一化尺寸
		\param[in] path 配置文件路径
	 */
	void init(const RegionOfInterest& observedRegion, const SizeInfo& sizesOrigAndNorm, 
        const std::string& path = std::string());
	//! 设置参数
	/*!
	    如果指针不为空, 则会对相关的内部参数进行更改
	    \param[in] allowedMissTimeInMinute 如果一个目标消失了最多这么多时间后重新在相同位置被发现, 则目标不会被删除, 而是继续被跟踪
		\param[in] minStaticTimeInMinute 如果被跟踪的目标的持续时间第一次达到这个值, 则会在 proc 函数中输出到 staticObjects 中
	 */
	void setConfigParam(const double* allowedMissTimeInMinute = 0, const double* minStaticTimeInMinute = 0);
	//! 处理函数
    /*!
        \param[in] time 时间戳
        \param[in] count 帧编号
        \param[in] rects 当前帧中的矩形
        \param[out] staticObjects 输出信息, 首次检测到静止状态的目标将会被输出
     */
	void proc(const long long int time, const int count, 
		const std::vector<cv::Rect>& rects, std::vector<StaticObjectInfo>& staticObjects);
	//! 画出目标
	/*!
	    \param[in,out] image 归一化尺寸输入图片
		\param[in] staticColor 已被判定为静止的目标的颜色
		\param[in] nonStaticColor 未被判定为静止的目标的颜色
	 */
	void drawBlobs(cv::Mat& image, const cv::Scalar& staticColor, const cv::Scalar& nonStaticColor) const;

private:
	StaticBlobTracker(const StaticBlobTracker&);
	StaticBlobTracker& operator=(const StaticBlobTracker&);

	class Impl;
	cv::Ptr<Impl> ptrImpl;
};

} // end namespace zsfo