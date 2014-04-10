#pragma once

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include "ExportControl.h"
#include "BlobTracker.h"

namespace zsfo
{

//! 加了时间戳和帧编号的图片
struct StampedImage
{
	//! 构造函数
    StampedImage(void) : time(0), number(0) {};

	long long int time;        ///< 时间戳
	int number;                ///< 帧编号
	cv::Mat image;             ///< 输入全景图
};

//! 描述检测到的目标的具体信息的结构体
struct ObjectDetails
{
    std::vector<ObjectInfo> objects;                ///< 输出的运动目标
	std::vector<StaticObjectInfo> staticObjects;    ///< 输出的静态目标
};

//! 跟踪和快照类
class Z_LIB_EXPORT MovingObjectDetector
{
public:
    //! 构造函数
    MovingObjectDetector(void) {};
    //! 根据配置文件进行初始化
    /*!
        \param[in] input 用于初始化的一帧和相关信息
        \param[in] pathPath 指明私有成员变量的配置文件路径的文件
     */
    void init(const StampedImage& input, const std::string& path);
    //! 根据函数传入参数进行初始化
    /*!
        部分参数为指针形式, 只要指针不为空指针, 就会将类的配置参数按指针指向的值进行设置
        \param[in] input 
                   用于初始化的一帧和相关信息
        \param[in] normSize 
                   视频的原始尺寸一般都比较大, 而且会有信息的冗余, 为了加快处理速度
                   将原始视频缩小至归一化尺寸进行处理, 一般取宽 320 高 240
        \param[in] updateBackInterval
                   调用多少次 proc 函数才进行一次背景模型的更新, 其他情况下只检测前景
        \param[in] recordMode 
                   历史信息存储模式
                   选择是仅保存矩形框跟踪历史, 还是附加抓拍图片历史
                   抓拍图片历史保存跨越线圈的左右下边界抓拍, 跨越线圈的下边界抓拍, 跨越线段抓拍, 和无线圈无线段的多图抓拍
        \param[in] saveMode
                   存图方式
                   选择是否保存全景图, 目标截图和目标前景图, 如果保存多幅图, 可采用位运算符将多个值相或
                   如果 recordMode == RecordMode::NoVisualRecord, 则此参数不起作用
        \param[in] saveInterval
                   仅当 recordMode == RecordMode::MultiVisualRecord 时起作用
                   当本类的 proc 函数被调用这么多次, 才保存一次历史截图
        \param[in] numOfSaved
                   仅当 recordMode == RecordMode::MultiVisualRecord 时起作用
                   最多保存的历史截图的数量
        \param[in] normScale 
                   指定后面的参数是在原始视频尺寸给出的还是归一化尺寸给出的
        \param[in] includeRegionPoints 
                   观测和跟踪区域的顶点坐标, 可以为单一线段, 一个或者多个多边形的组合
                   如果不给出, 默认为归一化尺寸矩形的四个顶点
        \param[in] excludeRegionPoints 
                   不进行跟踪和观测区域的顶点坐标, 可以为空, 可以为一个或者多个多边形的组合
                   includeRegionPoints 为空 excludeRegionPoints 不为空时, excludeRegionPoints 才会起作用 
        \param[in] crossLoopOrLineSegmentPoints 
                   抓拍图片用的线圈或者线段
                   如果是跨越线段抓拍, 必须给出两个点, 线段应当落在观测和跟踪线圈内部
                   如果是跨越底部边界抓拍, 本参数可以不给出, 默认采用一个比全图稍小一些的线圈
                   如果给出, 必须是 4 个点, 依次为左上角, 左下角, 右下角, 右上角
        \param[in] minObjectArea 
                   前景轮廓所包围的面积大于这个值才会被保留, 进行后续处理
        \param[in] minObjectWidth 
                   被允许的前景外接矩形的宽度的最小值
        \param[in] minObjectHeight 
                   被允许的前景外接矩形的高度的最小值
        \param[in] charRegionCheck 
                   是否检测前景矩形是否落在字符区域中, 如果检测, 落在字符区域中的前景矩形会被删除
        \param[in] charRegionRects 
                   框定字符区域的所有矩形组成的向量
        \param[in] checkTurnAround 
                   是否检测运动目标折返, 如果检测, 并且存在折返, 则删除当前跟踪目标, 建立新的跟踪目标
        \param[in] maxDistRectAndBlob 
                   如果当前帧某个矩形的中心和某个被跟踪对象在上一帧的矩形的中心距离小于这个值, 则满足匹配条件之一
        \param[in] minRatioIntersectToSelf 
                   如果当前帧某个矩形和某个被跟踪对象在上一帧的矩形的交集的面积和当前帧这个矩形的面积的比值大于这个值, 则满足匹配条件之一
        \param[in] minRatioIntersectToBlob 
                   如果当前帧某个矩形和某个被跟踪对象在上一帧的矩形的交集的面积和这个被跟踪对象矩形的面积的比值大于这个值, 则满足匹配条件之一
     */
    void init(const StampedImage& input, const cv::Size& normSize = cv::Size(320, 240), int updateBackInterval = 4,
        int recordMode = RecordMode::CrossTriBoundVisualRecord, int saveMode = SaveImageMode::SaveSlice, 
        int saveInterval = 2, int numOfSaved = 4, bool normScale = true, 
        const std::vector<std::vector<cv::Point> >& includeRegionPoints = std::vector<std::vector<cv::Point> >(),
        const std::vector<std::vector<cv::Point> >& excludeRegionPoints = std::vector<std::vector<cv::Point> >(),
        const std::vector<cv::Point>& crossLoopOrLineSegmentPoints = std::vector<cv::Point>(),
        const double* minObjectArea = 0, const double* minObjectWidth = 0, const double* minObjectHeight = 0,
        const bool* charRegionCheck = 0, const std::vector<cv::Rect>& charRegionRects = std::vector<cv::Rect>(),
        const bool* checkTurnAround = 0, const double* maxDistRectAndBlob = 0,
        const double* minRatioIntersectToSelf = 0, const double* minRatioIntersectToBlob = 0);
    //! 建立背景模型函数
    /*!
        只学习和更新背景模型, 不进行前景检测和跟踪
        \param[in] input 当前帧和时间戳, 帧编号等信息
     */
    void build(const StampedImage& input);
    //! 逐帧处理的函数
    /*!
        \param[in] input 当前帧和时间戳, 帧编号等信息
        \param[out] output 当前帧的处理结果, 包括正在跟踪的目标的当前信息, 和结束跟踪的目标的历史轨迹和快照
    */
	void proc(const StampedImage& input, ObjectDetails& output);
    //! 视频处理完之后调用 
    /*!
        视频已经处理完, 不管是否跟踪结束, 都将运动目标的历史轨迹和抓拍图片输出
     */
	void final(ObjectDetails& output);

private:
    class Impl;
    cv::Ptr<Impl> ptrImpl;
	MovingObjectDetector(const MovingObjectDetector& movObjDet);
	MovingObjectDetector& operator=(const MovingObjectDetector& movObjDet);
};

//! MovingObjectDetector 类输出结果解析类
class Z_LIB_EXPORT OutputInfoParser
{
public:
    //! 初始化
    /*!
        \param[in] savePath 图片和文本信息文件的保存路径
        \param[in] sceneImageName 全景图文件名的前缀, 若为空, 则不保存
        \param[in] sliceImageName 目标截图文件名的前缀, 若为空, 则不保存
        \param[in] maskImageName 目标前景图文件名的前缀, 若为空, 则不保存
        \param[in] objectInfoFileName 目标信息文件名, 若为空, 则不保存
        \param[in] objectHistoryFileName 目标历史轨迹文件名, 若为空, 则不保存
        \param[in] isPicSmall 原图尺寸是否小, 如果不是, 在调用 show 函数显示图片的时候, 会将图片的宽和高缩小至原来的一半再显示
        \param[in] waitKeyTime 显示图片后的等待响应时间
     */
	void init(const std::string& savePath, 
		const std::string& sceneImageName, const std::string& sliceImageName, const std::string& maskImageName,
		const std::string& objectInfoFileName, const std::string& objectHistoryFileName,
		bool isPicSmall = true, int waitKeyTime = 0);
    //! 新的视频帧或者图片分析完后, 调用该函数显示分析结果
    /*!
        显示处于跟踪状态的目标的信息, 对于结束跟踪的目标, 显示截图等
        \param[in] input 
        \param[in] output 
     */
	void show(const StampedImage& input, const ObjectDetails& output);
    //! 保存结束跟踪的目标的信息
	void save(const ObjectDetails& output);
    //! 视频或者图片序列结束后调用, 给文本文件加上结束符
    /*!
        \param[in] label 结束符
     */
	void final(const std::string& label = "End");

private:
    class Impl;
    cv::Ptr<Impl> ptrImpl;
};

//! 调用跟踪和快照类处理视频
/*!
    \param[in] videoName 需要处理的视频文件名, 如果为空, 会抛出 std::string 类型的异常
    \param[in] savePath 保存处理结果的路径, 如果为空, 会抛出 std::string 类型的异常
    \param[in] sceneName 保存的全景图的名称, savePath 文件夹中的全景图文件名格式为 sceneName123.jpg, 如果为空字符串, 则不保存全景图
    \param[in] sliceName 保存的目标截图的名称, savePath 文件夹中的目标截图文件名格式为 sliceName123.jpg, 如果为空字符串, 则不保存目标截图
    \param[in] maskName 保存的目标前景图的名称, savePath 文件夹中的目标前景图文件名格式为 maskName123.jpg, 如果为空字符串, 则不保存目标前景图
    \param[in] objectInfoFileName 保存的目标信息文件的名称, 如果为空字符串, 则不保存该文件
    \param[in] objectHistoryFileName 保存的历史轨迹文件的名称, 如果为空字符串, 则不保存该文件
    \param[in] procEveryNFrame 每这么多帧实际处理一帧, 如果小于 1, 会抛出 std::string 类型的异常
    \param[in] buildBackModelCount 建立背景模型需要实际处理的帧数, 包含初始化的那一帧
 */
void Z_LIB_EXPORT procVideo(const std::string& videoName, const std::string& savePath, 
    const std::string& sceneName = std::string(), const std::string& sliceName = std::string(), const std::string& maskName = std::string(), 
    const std::string& objectInfoFileName = std::string(), const std::string& objectHistoryFileName = std::string(),
    int procEveryNFrame = 1, const cv::Size& normSize = cv::Size(320, 240), 
    int buildBackModelCount = 20, int updateBackInterval = 4,
    int recordMode = RecordMode::CrossTriBoundVisualRecord, int saveMode = SaveImageMode::SaveSlice, 
    int saveInterval = 2, int numOfSaved = 4, bool normScale = true, 
    const std::vector<std::vector<cv::Point> >& includeRegionPoints = std::vector<std::vector<cv::Point> >(),
    const std::vector<std::vector<cv::Point> >& excludeRegionPoints = std::vector<std::vector<cv::Point> >(),
    const std::vector<cv::Point>& crossLoopOrLineSegmentPoints = std::vector<cv::Point>(),
    const double* minObjectArea = 0, const double* minObjectWidth = 0, const double* minObjectHeight = 0,
    const bool* charRegionCheck = 0, const std::vector<cv::Rect>& charRegionRects = std::vector<cv::Rect>(),
    const bool* checkTurnAround = 0, const double* maxDistRectAndBlob = 0,
    const double* minRatioIntersectToSelf = 0, const double* minRatioIntersectToBlob = 0);

}