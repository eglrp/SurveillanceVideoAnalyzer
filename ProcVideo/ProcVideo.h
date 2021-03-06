﻿#pragma once
#include <string>
#include <vector>
#include <utility>
#include "ExportControl.h"

namespace zpv
{

//! 任务信息
struct TaskInfo
{
    std::string taskID;            ///< 任务编号
    std::string videoPath;         ///< 视频全路径
    std::string videoSegmentID;    ///< 视频分段号
    std::pair<int, int> frameCountBegAndEnd; ///< 视频分段的起始帧号(包含)和结束帧号(包含)
    std::string saveImagePath;     ///< 保存图片的路径
    std::string saveHistoryPath;   ///< 保存历史轨迹文件的路径
    std::string historyFileName;   ///< 历史轨迹文件名, 最终历史轨迹文件是 saveHistoryPath\historyFileName
};

//! 场景类型(视距)
struct ZoomType
{
    enum 
    {
        SMALL_SCENE, 
        MIDDLE_SCENE, 
        LARGE_SCENE
    };
};

//! 场景类型(视角)
struct TiltType
{
    enum 
    {
        SMALL_ANGLE, 
        MIDDLE_ANGLE, 
        LARGE_ANGLE
    };
};

//! 场景类型(光照, 天气, 背景)
struct EnvironmentType
{
    enum 
    {
        SUNNY, 
        RAINY, 
        SNOWY, 
        CLOUDY,
        NIGHT
    };
};

//! 任务配置信息
struct ConfigInfo
{
    std::string configPath;  ///< 配置文件路径
    //! 感兴趣区域
    std::vector<std::vector<std::pair<int, int> > > includeRegion;
    //! 不感兴趣区域, 仅当感兴趣区域为空, 不感兴趣区域非空时才有效
    std::vector<std::vector<std::pair<int, int> > > excludeRegion;
    int tiltType;                ///< 视角类型
    int zoomType;                ///< 视距类型
    int environmentType;         ///< 光照天气背景类型
};

//! 跟踪对象信息
struct ObjectInfo
{
    int objectID;            ///< 目标编号
    //! 起始和结束的时间戳
    std::pair<long long int, long long int> timeBegAndEnd;   
    std::string sliceName;   ///< 全景图全路径
    std::string sceneName;   ///< 目标截图全路径
    int frameCount;          ///< 截图的帧编号
    struct Rect
    {
        int x;
        int y;
        int width;
        int height;
    };
    Rect sliceLocation;      ///< 截图在全景图中的位置
};

}

typedef void (*procVideoCallBack)(float progressPercentage, 
    const std::vector<zpv::ObjectInfo>& infos, void* ptrUserData);

namespace zpv
{
//! 分割视频函数
/*!
    可能会抛出 std::exception 类型的异常
    \param[in] videoPath 视频文件全路径
    \param[in] splitUnitInSecond 期望的每个视频分段长度, 以秒计算
    \param[out] videoLengthInSecond 视频的实际时长, 以秒计算
    \param[out] segmentLengthInSecond 每个视频分段的时长, 以秒计算
    \param[out] splitBegAndEnd 每个视频分段的起始帧号(包含)和结束帧号(包含), 帧号从 0 计数
    \return 函数成功执行, 返回 true, 否则返回 false
 */
Z_LIB_EXPORT bool findSplitPositions(const std::string& videoPath, const double splitUnitInSecond,
    double& videoLengthInSecond, std::vector<double>& segmentLengthInSecond,
    std::vector<std::pair<int, int> >& splitBegAndEnd);

//! 分割视频函数
/*!
    可能会抛出 std::exception 类型的异常
    \param[in] videoPath 视频文件全路径
    \param[in] expectSegNum 期望的视频分段
    \param[out] videoLengthInSecond 视频的实际时长, 以秒计算
    \param[out] segmentLengthInSecond 每个视频分段的时长, 以秒计算
    \param[out] splitBegAndEnd 每个视频分段的起始帧号(包含)和结束帧号(包含), 帧号从 0 计数
    \return 函数成功执行, 返回 true, 否则返回 false
 */
Z_LIB_EXPORT bool findSplitPositions(const std::string& videoPath, const int expectSegNum,
    double& videoLengthInSecond, std::vector<double>& segmentLengthInSecond,
    std::vector<std::pair<int, int> >& splitBegAndEnd);

//! 处理视频片段函数
/*!
    可能会抛出 std::exception 类型的异常
    \param[in] task 分析任务信息
    \param[in] config 配置信息
    \param[in] ptrCallBackFunc 回调函数指针
    \param[in,out] ptrUserData 用户数据
 */
Z_LIB_EXPORT void procVideo(const TaskInfo& task, const ConfigInfo& config, 
    procVideoCallBack ptrCallBackFunc, void* ptrUserData);
}

namespace zpv
{

//! 任务信息
struct TaicangTaskInfo
{
    std::string taskID;            ///< 任务编号
    std::string caseName;          ///< 案例名
    std::string caseSetName;       ///< 案例集合名
    std::string videoPath;         ///< 视频全路径
    std::string videoSegmentID;    ///< 视频分段号
    std::pair<int, int> frameCountBegAndEnd; ///< 视频分段的起始帧号(包含)和结束帧号(包含)
    std::string saveImagePath;     ///< 保存图片的路径
    std::string saveHistoryPath;   ///< 保存历史轨迹文件的路径
    std::string historyFileName;   ///< 历史轨迹文件名, 最终历史轨迹文件是 saveHistoryPath\historyFileName
};

//! 参数信息
struct TaicangParamInfo
{
    typedef std::pair<int, int> Size;
    typedef std::pair<int, int> Point;
    struct Rect
    {
        int x;
        int y;
        int width;
        int height;
    };
    Size normSize;          ///< 分析视频使用的归一化尺寸
    bool normScale;         ///< 后面的参数是使用归一化尺寸还是原始尺寸
    //! 感兴趣区域
    std::vector<std::vector<Point > > includeRegion;
    //! 不感兴趣区域, 仅当感兴趣区域为空, 不感兴趣区域非空时才有效
    std::vector<std::vector<Point > > excludeRegion;
    //! 过滤区域, 该区域内的物体会被过滤
    std::vector<Rect> filterRects;
    double minObjectArea;
    double minObjectWidth;
    double minObjectHeight;
    double maxMatchDist;
};

//! 跟踪对象信息
struct TaicangObjectInfo
{
    std::string taskID;      ///< 任务编号
    std::string caseName;    ///< 案例名
    std::string caseSetName; ///< 案例集合名
    int objectID;            ///< 目标编号
    //! 起始和结束的时间戳
    std::pair<long long int, long long int> timeBegAndEnd;   
    std::string sliceName;   ///< 全景图全路径
    std::string sceneName;   ///< 目标截图全路径
    int frameCount;          ///< 截图的帧编号
    struct Rect
    {
        int x;
        int y;
        int width;
        int height;
    };
    Rect sliceLocation;      ///< 截图在全景图中的位置
};
}

typedef void (*taicangProcVideoCallBack)(float progressPercentage, 
    const std::vector<zpv::TaicangObjectInfo>& infos, void* ptrUserData);

namespace zpv
{
//! 处理视频片段函数
/*!
    可能会抛出 std::exception 类型的异常
    \param[in] task 分析任务信息
    \param[in] param 参数信息
    \param[in] ptrCallBackFunc 回调函数指针
    \param[in,out] ptrUserData 用户数据
 */
Z_LIB_EXPORT void procVideo(const TaicangTaskInfo& task, const TaicangParamInfo& param,
    taicangProcVideoCallBack ptrCallBackFunc, void* ptrUserData);
}
