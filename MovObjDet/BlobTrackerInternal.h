#pragma once

#include <list>
#include "RegionOfInterest.h"
#include "MovingObjectDetector.h"
#include "BlobTracker.h"

namespace zsfo
{

//! 记录运动目标在一帧中的记录的结构体
struct BlobQuanRecord
{
    //! 构造函数
    BlobQuanRecord(void) : time(0), count(0) {};
    //! 根据输入信息生成记录
    /*!
        \param[in] rect 归一化尺寸的矩形
        \param[in] gradDiffMean 当前帧和上一帧在 rect 区域中的差值的平均值
        \param[in] time 时间戳
        \param[in] count 帧编号
        \param[in] sizeInfo 原始尺寸和归一化尺寸
     */
    void makeRecord(const cv::Rect& rect, double gradDiffMean, long long int time, int count, const SizeInfo& sizeInfo);

    cv::Rect rect;             ///< 矩形
    double gradDiffMean;       ///< 矩形区域梯度差的均值
    cv::Point top;             ///< 矩形顶部边界的中心
    cv::Point center;          ///< 矩形中心
    cv::Point bottom;          ///< 矩形底部边界的中心
	cv::Rect origRect;         ///< 原始帧中的矩形
    long long int time;        ///< 时间戳      
    int count;                 ///< 帧编号
};

//! 记录运动目标历史的结构体
struct BlobQuanHistory
{
    //! 构造函数
    /*!
        \param[in] sizesOrigAndNorm 原始尺寸和归一化尺寸
        \param[in] time 时间戳
        \param[in] count 帧编号
        \param[in] blobID 运动目标编号
     */
    BlobQuanHistory(const cv::Ptr<SizeInfo>& sizesOrigAndNorm, 
        const cv::Ptr<long long int>& time, const cv::Ptr<int>& count, int blobID);
    //! 拷贝构造函数, 只复制共享变量, 使用新的 ID
    /*!
        \param[in] history 拷贝构造新的实例时共享变量的来源
        \param[in] blobID 运动目标的编号
     */
    BlobQuanHistory(const BlobQuanHistory& history, int blobID);
    //! 根据现有实例拷贝构造创建新实例, 返回新实例的指针, 使用新的 ID 共享变量只增加引用计数 
    BlobQuanHistory* createNew(int blobID) const;
    //! 返回历史记录的长度
    int size(void) const;
	//! 根据当前帧得到的 rect 和 gradDiffMean 把记录 push 到向量中
    void pushRecord(const cv::Rect& rect, double gradDiffMean);
    //! 打印历史
    void displayHistory(void) const;
	//! 输出历史
	void outputHistory(ObjectInfo& objectInfo) const;
    //! 画当前的矩形
    void drawRect(cv::Mat& normalImage, const cv::Scalar& color) const;
    //! 画矩形中心点的历史轨迹
    void drawCenterHistory(cv::Mat& normalImage, const cv::Scalar& color) const;
    //! 画矩形上边中心点的历史轨迹
    void drawTopHistory(cv::Mat& normalImage, const cv::Scalar& color) const;
    //! 画矩形下边中心点的历史轨迹
    void drawBottomHistory(cv::Mat& normalImage, const cv::Scalar& color) const;
    //! 获取矩形中心点的历史轨迹
    void getCenterHistory(std::vector<cv::Point>& centerHistory) const;
    //! 检测矩形区域和相应的纹理是否稳定
    bool checkStability(int timeInMilliSec) const;
	//! 对矩形中心拟合轨迹
    /*!
        \param[out] pointInLine 拟合直线上的一点
        \param[out] dirVector 拟合直线的单位方向向量
        \param[out] avgError 用于拟合直线的点到拟合直线的距离的平均值
     */
	void linearRegres(cv::Point& pointInLine, cv::Point2d& dirVector, double& avgError) const;
	//! 检测是否有掉头
	bool checkTurnAround(void) const;
	//! 检测 Y 轴方向上的运动方向是否和合法方向一致
	bool checkYDirection(int legalDirection) const;
	
    std::vector<BlobQuanRecord> history;       ///< 矩形各类信息的历史记录
	BlobQuanRecord initRecord;                 ///< 初始帧矩形记录
    BlobQuanRecord currRecord;                 ///< 当前帧矩形记录

	// X 轴和 Y 轴运动方向的向量
	// if currCenter.x > lastCenter.x + maxDiffVal, sign = 1
	// else if currCenter.x < lastCenter.x - maxDiffVal, sign = -1
	// else sign = 0
	std::vector<char> dirCenterX;  ///< 矩形中心点 x 坐标的运动方向
	std::vector<char> dirCenterY;  ///< 矩形中心点 y 坐标的运动方向
    // 以下两者为共享变量
	int checkDirStep;       ///< 间隔多少帧检测一次矩形中心点的运动方向
	int maxDiffVal;         ///< 判定运动方向的阈值

	int ID;                             ///< 运动目标的编号
    // 共享变量
	cv::Ptr<SizeInfo> sizeInfo;         ///< 原始尺寸和归一化尺寸
    cv::Ptr<long long int> currTime;    ///< 时间戳
    cv::Ptr<int> currCount;             ///< 帧编号
};

//! 全景图延迟处理类
class OrigSceneProxy
{
public:
    OrigSceneProxy(const cv::Mat& frame) : done(false), shallowCopy(frame) {};
    cv::Mat getDeepCopy(void);
private:    
    cv::Mat shallowCopy;
    cv::Mat deepCopy;
    bool done;
};

//! 保存抓拍图片和相关信息的结构体
struct BlobVisualRecord
{
	//! 构造函数
    BlobVisualRecord(void) : bound(-1), crossIn(-1), direction(-1), time(0), count(0) {};
    //! 根据输入信息创建记录结构体
    /*!
        \param[in] origFrame 原始尺寸的输入图片, 即全景图
        \param[in] normForeImage 归一化尺寸的全景图
        \param[in] sizeInfo 原始尺寸和归一化尺寸
        \param[in] baseRect 归一化尺寸的截图矩形区域, 该矩形以外的区域不截图
        \param[in] blobRect 归一化尺寸的运动目标的矩形
        \param[in] currTime 时间戳
        \param[in] currCount 帧编号
        \param[in] saveMode 存图模式, 选择是否保存全景图, 目标截图和目标前景图
     */
    void makeRecord(const cv::Mat& origFrame, const cv::Mat& normForeImage, 
		const SizeInfo& sizeInfo, const cv::Rect& baseRect, const cv::Rect& blobRect, 
        long long int currTime, int currCount, int saveMode);
    //! 根据输入信息创建记录结构体
    /*!
        \param[in] origFrame 原始尺寸的输入图片, 即全景图
        \param[in] normForeImage 归一化尺寸的全景图
        \param[in] sizeInfo 原始尺寸和归一化尺寸
        \param[in] baseRect 归一化尺寸的截图矩形区域, 该矩形以外的区域不截图
        \param[in] blobRect 归一化尺寸的运动目标的矩形
        \param[in] loopBound 获得截图时跨越的线圈的边界
        \param[in] crossMode 获得截图时目标的运动方向
        \param[in] currTime 时间戳
        \param[in] currCount 帧编号
        \param[in] saveMode 存图模式, 选择是否保存全景图, 目标截图和目标前景图
     */
    void makeRecord(const cv::Mat& origFrame, const cv::Mat& normForeImage, 
		const SizeInfo& sizeInfo, const cv::Rect& baseRect, const cv::Rect& blobRect, 
        int loopBound, int crossMode, long long int currTime, int currCount, int saveMode);
    //! 深拷贝复制 record 得到新的实例
    void copyTo(BlobVisualRecord& record) const;
    //! 将时间和在原始尺寸上的图输出到对应的结构体中
	void outputImages(ObjectVisualRecord& visualRecord) const;

    int bound;                   ///< 跨越的是线圈的哪个边界 1 跨越线圈左边界 2 跨越线圈右边界 3 跨越线圈下边界 -1 跨越检测线
	int crossIn;                 ///< 抓拍时是否为进入线圈 1 是 0 不是 -1 未知
	int direction;               ///< 抓拍时的行驶方向 1 从左到右 2 从右到左 3 从上到下 4 从下到上 -1 未知
    cv::Rect normRect;           ///< 归一化帧上的矩形
    cv::Rect origRect;           ///< 原始帧上的矩形
    long long int time;          ///< 时间戳
	int count;                   ///< 帧编号
    cv::Mat blobImage;           ///< 原始尺寸的运动目标图片, 大小等于 origRect 的大小
    cv::Mat foreImage;           ///< 原始尺寸的运动目标前景图, 大小等于 origRect 的大小
    cv::Mat fullFrame;           ///< 原始尺寸的全景图
};

//! 抓拍图片基类
struct BlobVisualHistory
{
    //! 析构函数
    virtual ~BlobVisualHistory(void) {};
    //! 根据本实例创建一个新的实例, 分配新的编号 blobID, 只拷贝共享变量
    virtual BlobVisualHistory* createNew(int blobID) const = 0;
    //! 更新图像记录历史
    /*!
        \param[in] origFrame 原始尺寸的输入图片
        \param[in] foreImage 归一化尺寸的全景图
        \param[in] currRect 归一化尺寸下运动目标的矩形
     */
    virtual void updateHistory(const cv::Mat& origFrame, const cv::Mat& foreImage, const cv::Rect& currRect) = 0;
    //! 输出图片历史到 objectInfo 中
    virtual bool outputHistory(ObjectInfo& objectInfo) const = 0;
};

//! 记录跨越三条线的抓拍图片的结构体
struct BlobTriBoundVisualHistory : public BlobVisualHistory
{
    //! 构造函数
    /*!
        \param[in] catchLoop 抓拍图片使用的线圈, 矩形边跨越线圈左边界或者右边界或者下边界才进行抓拍
        \param[in] sizesOrigAndNorm 原始尺寸和归一化尺寸
        \param[in] boundRect 归一化尺寸的基准矩形, 落在基准矩形内的部分才会保存
        \param[in] time 时间戳
        \param[in] count 帧编号
        \param[in] blobID 运动目标编号
        \param[in] saveMode 存图模式, 选择是否保存全景图, 目标截图和目标前景图
        \param[in] path 配置文件路径
     */
    BlobTriBoundVisualHistory(const cv::Ptr<VirtualLoop>& catchLoop, const cv::Ptr<SizeInfo>& sizesOrigAndNorm, 
		const cv::Ptr<cv::Rect>& boundRect, const cv::Ptr<long long int>& time, const cv::Ptr<int>& count, 
        int blobID, int saveMode, const std::string& path = std::string());  
    //! 拷贝构造函数只复制共享变量
    BlobTriBoundVisualHistory(const BlobTriBoundVisualHistory& history, int blobID);
    //! 虚析构
    virtual ~BlobTriBoundVisualHistory() {};
    //! 根据现有实例拷贝构造新的实例 使用新的 ID 共享变量只增加引用计数 
    virtual BlobTriBoundVisualHistory* createNew(int blobID) const;      
	//! 更新记录历史
	virtual void updateHistory(const cv::Mat& origFrame, const cv::Mat& foreImage, const cv::Rect& currRect);
	//! 输出图片记录
	virtual bool outputHistory(ObjectInfo& objectInfo) const;

	int ID;                                ///< 运动目标编号
	bool hasLeftCrossLoopLeft;             ///< 矩形左边是否已经跨越线圈左边
	bool hasRightCrossLoopRight;           ///< 矩形右边是否已经跨越线圈右边	
	bool hasBottomCrossLoopBottom;         ///< 矩形底边是否已经跨越线圈底边
	BlobVisualRecord leftRecord;           ///< 矩形左边跨越线圈左边时的记录
	BlobVisualRecord rightRecord;          ///< 矩形右边跨越线圈右边时的记录	
	BlobVisualRecord bottomRecord;         ///< 矩形底边跨越线圈底边时的记录
    BlobVisualRecord auxiRecord;           ///< 辅助记录 永远是跨越底部边界 进线圈在没有边界记录的时候会输出
	
    cv::Ptr<VirtualLoop> recordLoop;       ///< 抓拍图片的线圈
	cv::Ptr<SizeInfo> sizeInfo;            ///< 原始尺寸和归一化尺寸
    cv::Ptr<cv::Rect> baseRect;            ///< 截图的基准矩形
    cv::Ptr<long long int> currTime;       ///< 时间戳
    cv::Ptr<int> currCount;                ///< 帧编号
	bool hasUpdate;                        ///< 是否已经保存了至少一次记录
	cv::Rect lastRect;                     ///< 上一帧中的归一化只存矩形
	int auxiCount;                         ///< auxiRecord 的辅助计数

    //! updateHistory 函数的配置参数
    struct ConfigUpdate
    {
        bool runShowImage;  ///< 是否在抓拍到图片时显示
        int waitTime;       ///< 显示图片的停留时间
        int saveMode;       ///< 保存哪些图片
    };
    cv::Ptr<ConfigUpdate> configUpdate;     ///< 保存配置参数
};

//! 记录跨越底边条线的抓拍图片的结构体
struct BlobBottomBoundVisualHistory : public BlobVisualHistory
{
    // 构造函数
    /*!
        \param[in] catchLoop 抓拍图片使用的线圈, 矩形边跨越线圈下边界才进行抓拍
        \param[in] sizesOrigAndNorm 原始尺寸和归一化尺寸
        \param[in] boundRect 归一化尺寸的基准矩形, 落在基准矩形内的部分才会保存
        \param[in] time 时间戳
        \param[in] count 帧编号
        \param[in] blobID 运动目标编号
        \param[in] saveMode 存图模式, 选择是否保存全景图, 目标截图和目标前景图
        \param[in] path 配置文件路径
     */
    BlobBottomBoundVisualHistory(const cv::Ptr<VirtualLoop>& catchLoop, const cv::Ptr<SizeInfo>& sizesOrigAndNorm, 
		const cv::Ptr<cv::Rect>& boundRect, const cv::Ptr<long long int>& time, const cv::Ptr<int>& count, 
        int blobID, int saveMode, const std::string& path = std::string());
    // 拷贝构造函数只复制共享变量
    BlobBottomBoundVisualHistory(const BlobBottomBoundVisualHistory& history, int blobID);
    // 虚析构
    virtual ~BlobBottomBoundVisualHistory() {};
    // 根据现有实例拷贝构造新的实例, 使用新的 ID, 共享变量只增加引用计数 
    virtual BlobBottomBoundVisualHistory* createNew(int blobID) const;
	// 更新记录历史
	virtual void updateHistory(const cv::Mat& origFrame, const cv::Mat& foreImage, const cv::Rect& currRect);
	// 输出图片记录
	virtual bool outputHistory(ObjectInfo& objectInfo) const;

	int ID;                                ///< 运动目标编号
	bool hasBottomCrossLoopBottom;         ///< 矩形底边是否已经跨越线圈底边
	BlobVisualRecord bottomRecord;         ///< 矩形底边跨越线圈底边时的记录
	BlobVisualRecord auxiRecord;           ///< 辅助记录 永远是跨越底部边界 进线圈在没有边界记录的时候会输出

    cv::Ptr<VirtualLoop> recordLoop;        ///< 抓拍图片的线圈
	cv::Ptr<SizeInfo> sizeInfo;            ///< 原始尺寸和归一化尺寸
    cv::Ptr<cv::Rect> baseRect;            ///< 截图的基准矩形
    cv::Ptr<long long int> currTime;       ///< 时间戳
    cv::Ptr<int> currCount;                ///< 帧编号
	bool hasUpdate;                        ///< 是否已经保存了至少一次记录
	cv::Rect lastRect;                     ///< 上一帧中的归一化只存矩形
	int auxiCount;                         ///< auxiRecord 的辅助计数

    //! updateHistory 函数的配置参数
    struct ConfigUpdate
    {
        bool runShowImage;  ///< 是否在抓拍到图片时显示
        int waitTime;       ///< 显示图片的停留时间
        int saveMode;       ///< 保存哪些图片
    };
    cv::Ptr<ConfigUpdate> configUpdate;     ///< 保存配置参数
};

//! 记录跨越任意直线的抓拍图片的结构体
struct BlobCrossLineVisualHistory : public BlobVisualHistory
{
    //! 构造函数
    /*!
        \param[in] lineToCross 抓拍图片使用的线段, 当矩形中心靠近线圈时才进行抓拍
        \param[in] sizesOrigAndNorm 原始尺寸和归一化尺寸
        \param[in] boundRect 归一化尺寸的基准矩形, 落在基准矩形内的部分才会保存
        \param[in] time 时间戳
        \param[in] count 帧编号
        \param[in] blobID 运动目标编号
        \param[in] saveMode 存图模式, 选择是否保存全景图, 目标截图和目标前景图
        \param[in] path 配置文件路径
     */
    BlobCrossLineVisualHistory(const cv::Ptr<LineSegment>& lineToCross, const cv::Ptr<SizeInfo>& sizesOrigAndNorm, 
		const cv::Ptr<cv::Rect>& boundRect, const cv::Ptr<long long int>& time, const cv::Ptr<int>& count, 
        int blobID, int saveMode, const std::string& path = std::string());
    //! 拷贝构造函数只复制共享变量
    BlobCrossLineVisualHistory(const BlobCrossLineVisualHistory& history, int blobID);
    //! 虚析构
    virtual ~BlobCrossLineVisualHistory() {};  
    //! 根据现有实例拷贝构造新的实例 使用新的 ID 共享变量只增加引用计数 
    virtual BlobCrossLineVisualHistory* createNew(int blobID) const;
	//! 更新记录历史
	virtual void updateHistory(const cv::Mat& origFrame, const cv::Mat& foreImage, const cv::Rect& currRect);
	//! 输出图片记录
	virtual bool outputHistory(ObjectInfo& objectInfo) const;

	int ID;                                ///< 运动目标编号
	bool hasCrossLine;                     ///< 矩形底边是否已经跨越线段
	BlobVisualRecord crossLineRecord;      ///< 矩形底边跨越线段时的记录
	
    cv::Ptr<LineSegment> recordLine;       ///< 线段
	cv::Ptr<SizeInfo> sizeInfo;            ///< 原始尺寸和归一化尺寸
    cv::Ptr<cv::Rect> baseRect;            ///< 截图的基准矩形
    cv::Ptr<long long int> currTime;       ///< 时间戳
    cv::Ptr<int> currCount;                ///< 帧编号
	bool hasUpdate;                        ///< 是否已经保存了至少一次记录
	int lastMinDist;                       ///< 现存记录的矩形中心到线段的距离
    int maxDistToRecord;                   ///< 当矩形中心点到线段的距离小于这个值时才保存抓拍记录
	int auxiCount;                         ///< auxiRecord 的辅助计数

    //! updateHistory 函数的配置参数
    struct ConfigUpdate
    {
        bool runShowImage;  ///< 是否在抓拍到图片时显示
        int waitTime;       ///< 显示图片的停留时间
        int saveMode;       ///< 保存哪些图片
    };
    cv::Ptr<ConfigUpdate> configUpdate;     ///< 保存配置参数
};

//! 保存多张图片历史记录的结构体
struct BlobMultiRecordVisualHistory : public BlobVisualHistory
{
    //! 构造函数
    /*!
        \param[in] sizesOrigAndNorm 原始尺寸和归一化尺寸
        \param[in] boundRect 归一化尺寸的基准矩形, 落在基准矩形内的部分才会保存
        \param[in] time 时间戳
        \param[in] count 帧编号
        \param[in] blobID 运动目标编号
        \param[in] saveMode 存图模式, 选择是否保存全景图, 目标截图和目标前景图
        \param[in] saveInterval 保存图片的帧间隔
        \param[in] numOfSaved 最多保存多少张图片
        \param[in] path 配置文件路径
     */
    BlobMultiRecordVisualHistory(const cv::Ptr<SizeInfo>& sizesOrigAndNorm, const cv::Ptr<cv::Rect>& boundRect, 
        const cv::Ptr<long long int>& time, const cv::Ptr<int>& count, 
        int blobID, int saveMode, int saveInterval, int numOfSaved, const std::string& path = std::string());
    //! 拷贝构造函数只复制共享变量
    BlobMultiRecordVisualHistory(const BlobMultiRecordVisualHistory& history, int blobID);
    //! 虚析构
    virtual ~BlobMultiRecordVisualHistory() {};  
    //! 根据现有实例拷贝构造新的实例 使用新的 ID 共享变量只增加引用计数 
    virtual BlobMultiRecordVisualHistory* createNew(int blobID) const;
	//! 更新记录历史
	virtual void updateHistory(const cv::Mat& origFrame, const cv::Mat& foreImage, const cv::Rect& currRect);
	//! 输出图片记录
	virtual bool outputHistory(ObjectInfo& objectInfo) const;

	int ID;                                ///< 运动目标编号
    std::vector<BlobVisualRecord> history; ///< 保存的记录
    int auxiCount;                         ///< 辅助保存记录的计数
    bool allInside;                        ///< 所有历史记录都不与画面边界重合的标志

	cv::Ptr<SizeInfo> sizeInfo;            ///< 原始尺寸和归一化尺寸
    cv::Ptr<cv::Rect> baseRect;            ///< 截图的基准矩形
    cv::Ptr<long long int> currTime;       ///< 时间戳
    cv::Ptr<int> currCount;                ///< 帧编号

    //! updateHistory 函数的配置参数
    struct ConfigUpdate
    {
        bool runShowImage;  ///< 是否在抓拍到图片时显示
        int waitTime;       ///< 显示图片的停留时间
        int saveMode;       ///< 保存哪些图片
        int saveInterval;   ///< 保存图片的帧间隔
        int numOfSaved;     ///< 最多保存多少张图片
    };
    cv::Ptr<ConfigUpdate> configUpdate;     ///< 保存配置参数
};

//! 运动目标
class Blob
{
public:
    //! 只保存矩形历史的构造函数
    /*!
        \param[in] sizesOrigAndNorm 原始尺寸和归一化尺寸
        \param[in] time 时间戳
        \param[in] count 帧编号
        \param[in] blobID 运动目标编号
        \param[in] path 配置文件路径
     */
    Blob(const cv::Ptr<SizeInfo>& sizesOrigAndNorm, 
        const cv::Ptr<long long int>& time, const cv::Ptr<int>& count, int blobID, const std::string& path = std::string());
    //! 跨线抓拍使用的构造函数
    /*!
        \param[in] crossLine 抓拍图片使用的线段, 当矩形中心靠近线圈时才进行抓拍
        \param[in] sizesOrigAndNorm 原始尺寸和归一化尺寸
        \param[in] baseRect 归一化尺寸的基准矩形, 落在基准矩形内的部分才会保存
        \param[in] time 时间戳
        \param[in] count 帧编号
        \param[in] blobID 运动目标编号
        \param[in] saveMode 存图模式, 选择是否保存全景图, 目标截图和目标前景图
        \param[in] path 配置文件路径
     */
    Blob(const cv::Ptr<LineSegment>& crossLine, const cv::Ptr<SizeInfo>& sizesOrigAndNorm, 
        const cv::Ptr<cv::Rect>& baseRect, const cv::Ptr<long long int>& time, const cv::Ptr<int>& count, 
        int blobID, int saveMode, const std::string& path = std::string());
    //! 跨越底部边界和左边右边底边三条边界抓拍使用的构造函数
    /*!
        \param[in] catchLoop 抓拍图片使用的线圈
        \param[in] sizesOrigAndNorm 原始尺寸和归一化尺寸
        \param[in] baseRect 归一化尺寸的基准矩形, 落在基准矩形内的部分才会保存
        \param[in] time 时间戳
        \param[in] count 帧编号
        \param[in] blobID 运动目标编号
        \param[in] isTriBound 是跨越线圈左右下三边界抓拍还是跨越底边界才抓拍
        \param[in] saveMode 存图模式, 选择是否保存全景图, 目标截图和目标前景图
        \param[in] path 配置文件路径
     */
    Blob(const cv::Ptr<VirtualLoop>& catchLoop, const cv::Ptr<SizeInfo>& sizesOrigAndNorm, 
        const cv::Ptr<cv::Rect>& baseRect, const cv::Ptr<long long int>& time, const cv::Ptr<int>& count, 
        int blobID, bool isTriBound, int saveMode, const std::string& path = std::string());
    //! 保存多幅历史图片的构造函数
    /*!
        \param[in] sizesOrigAndNorm 原始尺寸和归一化尺寸
        \param[in] time 时间戳
        \param[in] count 帧编号
        \param[in] blobID 运动目标编号
        \param[in] saveMode 存图模式, 选择是否保存全景图, 目标截图和目标前景图
        \param[in] saveInterval 保存图片的帧间隔
        \param[in] numOfSaved 最多保存多少张图片
        \param[in] path 配置文件路径
     */
    Blob(const cv::Ptr<SizeInfo>& sizesOrigAndNorm, const cv::Ptr<cv::Rect>& baseRect, 
        const cv::Ptr<long long int>& time, const cv::Ptr<int>& count, 
        int blobID, int saveMode, int saveInterval, int numOfSaved, const std::string& path = std::string());
    //! 拷贝构造函数 历史记录则重新分配内存 所有共享变量只增加引用计数 
    Blob(const Blob& blob, int blobID, const cv::Rect& rect);
    //! 析构函数
	~Blob(void);
    //! 创建一个新的实例
    Blob* createNew(int blobID, const cv::Rect& rect) const;
	//! 更新运动目标状态, 包括更新历史记录, 仅适用于不保存图片的跟踪模式
	void updateState(void);
	//! 更新运动目标状态, 包括更新历史记录
    /*!
        \param[in] origFrame 原始尺寸的视频帧
        \param[in] foreImage 归一化尺寸的前景图
     */
	void updateState(const cv::Mat& origFrame, const cv::Mat& foreImage);
    //! 更新运动目标状态, 包括更新历史记录
    /*!
        \param[in] origFrame 原始尺寸的视频帧
        \param[in] foreImage 归一化尺寸的前景图
        \param[in] gradDiffImage 归一化尺寸输入图和背景图的梯度图的差值图
        \param[in] lastGradDiffImage 上一次处理时得到的归一化尺寸输入图和背景图的梯度图的差值图
     */
    void updateState(const cv::Mat& origFrame, const cv::Mat& foreImage, 
        const cv::Mat& gradDiffImage, const cv::Mat& lastGradDiffImage);	
	//! 输出运动目标的图片和属性
    /*!
        \param[out] output 运动目标信息写入的结构体
        \param[in] isFinal 是否视频已经处理结束, 如果是, 则仍处于跟踪状态的目标会输出历史轨迹和抓拍图片
     */
	bool outputInfo(ObjectInfo& objectInfo, bool isFinal) const;
    //! 画矩形
    void drawBlob(cv::Mat& normalImage, const cv::Scalar& color) const;
    //! 画历史
    void drawHistory(cv::Mat& normalImage, const cv::Scalar& color) const;
    //! 获取 Blob 的 ID
    int getID(void) const;
    //! 获取 Blob 的当前矩形位置
    cv::Rect getCurrRect(void) const;
    //! 获取是否处于被删除的状态
    bool getIsToBeDeleted(void) const;
    //! 获取 Blob 历史记录的长度
    int getHistoryLength(void) const;
    //! 获取 Blob 矩形历史记录的中心点的历史记录
    void getCenterHistory(std::vector<cv::Point>& centerHistory) const;
    //! 设置 Blob 当前矩形位置
    void setCurrRect(const cv::Rect& rect);
    //! 设置 Blob 将要被删除
    void setToBeDeleted(void);
    //! 检测 Blob 的轨迹是否有折返
    bool doesTurnAround(void) const;
    //! 打印 Blob 的历史
    void printHistory(void) const;

private:
    //! 未实现的拷贝构造函数
    Blob(const Blob& blob);
    //! 未实现的赋值符号
    Blob& operator=(const Blob& blob);
    //! 初始化配置参数 各配置参数结构体需要预先分配内存
    void initConfigParam(const std::string& path = std::string());

    int ID;				                       ///< 记录当前的 ID 号
    cv::Rect matchRect;                        ///< 当前帧中的矩形
	bool isToBeDeleted;                        ///< 如果等于 true，在 updateBlobListAfterCheck 中进行删除
	cv::Ptr<BlobQuanHistory> rectHistory;      ///< 记录矩形的历史
    cv::Ptr<BlobVisualHistory> visualHistory;  ///< 记录抓拍图片

	//! outputInfo 函数配置参数
	struct ConfigOutputInfo
	{
		int minHistorySizeForOutput;            ///< 历史轨迹长度超过这个值, 才会在跟踪结束时输出历史和图片
		bool runOutputHistory;                  ///< 是否输出历史轨迹
		bool runOutputVisualAndState;           ///< 是否输出图片
	};
	cv::Ptr<ConfigOutputInfo> configOutputInfo; ///< outputInfo 函数配置参数
};

//! 管理所有跟踪对象和所有共享资源的类
class BlobTracker::BlobTrackerImpl
{
public:
    //! 构造函数
    BlobTrackerImpl(void) {};
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
        \param[in] time 时间戳
        \param[in] count 帧编号
        \param[in] rects 当前帧中检测到的矩形
        \param[out] objects 输出信息, 当前处于跟踪状态的目标输出位置等信息, 结束跟踪的目标输出历史轨迹
     */
    void proc(long long int time, int count, const std::vector<cv::Rect>& rects, std::vector<ObjectInfo>& objects);
	//! 处理函数
    /*!
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
    BlobTrackerImpl(const BlobTrackerImpl&);
    BlobTrackerImpl& operator=(const BlobTrackerImpl&);

    //! 初始化配置参数
    /*!
        \param[in] path 配置文件路径
     */
    void initConfigParam(const std::string& path = std::string());
	//! blobs 链表更新函数
    /*!
        当前帧的矩形和 blobs 链表中的目标进行匹配, 未被匹配的矩形新建跟踪目标, blobs 中无法匹配的矩形标记为被删除
        \param[in] rects 当前帧中检测到的矩形
        \param[in] origFrame 原始尺寸输入帧
        \param[in] foreImage 归一化尺寸前景图
     */
	void updateBlobListBeforeCheck(const std::vector<cv::Rect>& rects);
    //! blobs 链表更新函数
    /*!
        删除 blobs 中被标记为删除的运动目标
     */
	void updateBlobListAfterCheck(void);
	//! 更新运动目标状态
	void updateState(void);
	void updateState(const cv::Mat& origFrame, const cv::Mat& foreImage);
    void updateState(const cv::Mat& origFrame, const cv::Mat& foreImage, 
        const cv::Mat& gradDiffImage, const cv::Mat& lastGradDiffImage);
    //! 画跟踪框和 ID 
	void drawObjects(cv::Mat& frame, const cv::Scalar& color) const;
	//! 画每个跟踪框的历史
	void drawHistories(cv::Mat& frame, const cv::Scalar& color) const;
	//! 在 blobList 中添加新的运动目标
	void addBlob(const cv::Rect& rect);
	//! 匹配函数
	void match(const std::vector<cv::Rect>& rects);
    //! 输出运动目标的图片和属性
	bool outputInfo(std::vector<ObjectInfo>& objects, bool isFinal = false) const;

	std::list<cv::Ptr<Blob> > blobList;///< 检测出的运动目标都放在这个结构体中
    cv::Ptr<Blob> blobInstance;        ///< 一个 Blob 实例，用于拷贝构造新的 Blob 实例
	int blobCount;                     ///< 总的 blob 个数

	// 执行 init 函数后 以下引用计数指针分配内存
    // 给本实例所管理的 Blob 实例共享
    cv::Ptr<RegionOfInterest> roi;     ///< 观测区域
    cv::Ptr<VirtualLoop> recordLoop;   ///< 抓拍图片的线圈
    cv::Ptr<LineSegment> recordLine;   ///< 跨线检测中的线圈
	cv::Ptr<long long int> currTime;   ///< 当前时间, 单位为毫秒
	cv::Ptr<int> currCount;            ///< 当前帧编号
	cv::Ptr<SizeInfo> sizeInfo;        ///< 原始尺寸和归一化尺寸
    cv::Ptr<cv::Rect> baseRect;        ///< 抓拍图片时使用的基准矩形, 图片只取落在基准矩形内的部分

	//! match 函数的配置参数
    struct ConfigMatch
	{
		bool runCheckTurnAround;          ///< 是否检测运动目标折返, 如果检测, 并且存在折返, 则删除当前跟踪目标, 建立新的跟踪目标
		double maxDistRectAndBlob;        ///< 如果当前帧某个矩形的中心和某个被跟踪对象在上一帧的矩形的中心距离小于这个值, 则满足匹配条件之一
		double minRatioIntersectToSelf;   ///< 如果当前帧某个矩形和某个被跟踪对象在上一帧的矩形的交集的面积和当前帧这个矩形的面积的比值大于这个值, 则满足匹配条件之一
		double minRatioIntersectToBlob;   ///< 如果当前帧某个矩形和某个被跟踪对象在上一帧的矩形的交集的面积和这个被跟踪对象矩形的面积的比值大于这个值, 则满足匹配条件之一
		int maxHistorySizeForDistMatch;   ///< 如果被跟踪对象的历史长度小于这个值, 不用直线拟合的方式进行匹配
		double maxAvgErrorForDistMatch;   ///< 矩形中心历史进行拟合后, 所有中心点到直线的距离的平均值小于这个值, 采根据拟合直线得到的结果进行匹配
		bool runDisplayCalcResults;       ///< 显示匹配过程中计算的数据
		bool runShowFitLine;              ///< 显示拟合得到的直线
	};
	ConfigMatch configMatch;              ///< match 函数配置参数实例
};

}