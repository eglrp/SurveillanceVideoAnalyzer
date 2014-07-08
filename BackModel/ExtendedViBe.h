#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include "ExportControl.h"

namespace zsfo
{

/*!
    基于存储历史像素样本值和邻域样本值更新策略的前景提取算法
    ViBe: A Universal Background Subtraction Algorithm for Video Sequences
    in IEEE Transactions on Image Processing, June, 2011, Volume 20, Issue 6
 */
class ViBe
{
public:
    //! 配置参数, 结构体中的参数的含义参见论文
    struct Config
    {
        //! 获取处理彩色图的参数
        static Config getRGBConfig(void)
        {
            return Config("[rgb]", 20, 40, 2, 16);
        }
        //! 获取处理灰度图的参数
        static Config getGrayConfig(void)
        {
            return Config("[gray]", 20, 10, 2, 16);
        }
        //! 获取处理梯度图的参数
        static Config getGradientConfig(void)
        {
            return Config("[gradient]", 20, 40, 2, 16);
        }
        //! 构造函数
        Config(const std::string& label_, int numOfSamples_, int minMatchDist_, int minNumOfMatchCount_, int subSampleInterval_)
            : label(label_), numOfSamples(numOfSamples_), minMatchDist(minMatchDist_), 
              minNumOfMatchCount(minNumOfMatchCount_), subSampleInterval(subSampleInterval_)
        {}
        std::string label;       ///< 标签
        int numOfSamples;        ///< 每个像素保存的样本数量
        int minMatchDist;        ///< 处理图片的高度
        int minNumOfMatchCount;  ///< 判定为背景的最小匹配成功次数
        int subSampleInterval;   ///< 它的倒数等于更新保存像素值的概率        
    };
    //! 初始化模型
    /*!
        传入第一帧画面 image, 给定配置参数 config
        image 必须是 CV_8UC1 或者 CV_8UC3 格式, 否则会抛出 std::exception 类型的异常
     */
	Z_LIB_EXPORT void init(const cv::Mat& image, const Config& config);
    //! 提取前景, 更新模型
    /*!
        结合现有模型参数, 检测输入图片 image 中的前景, 输出到 foregroundImage 中
        image 的尺寸和格式必须和 init 函数中进行初始化的图片的尺寸和格式完全相同, 否则会抛出 std::exception 类型的异常
        foregroundImage 的尺寸和 image 相同, 格式为 CV_8UC1, 前景像素值等于 255, 背景像素值等于 0
        rectsNoUpdate 中指定的矩形区域只进行前景检测, 不更新背景模型
     */
	Z_LIB_EXPORT void update(const cv::Mat& image, cv::Mat& foregroundImage,
		const std::vector<cv::Rect>& rectsNoUpdate = std::vector<cv::Rect>());
    //! 重置背景模型
    /*!
        使用 image 重置背景模型
        image 的尺寸和格式必须和 init 函数中进行初始化的图片的尺寸和格式完全相同, 否则会抛出 std::exception 类型的异常
     */
	Z_LIB_EXPORT void refill(const cv::Mat& image);
    //! 显示每个像素的背景样本值
    /*!
        函数显示每个像素存储的前 count 个背景样本的值
        如果 count 小于等于 0 或者大于实际背景样本的数量, 则不进行任何操作
     */
    Z_LIB_EXPORT void showSamples(int count);

protected:
	int imageWidth;                         ///< 处理图片的宽度
    int imageHeight;                        ///< 处理图片的高度
    cv::Rect imageRect;                     ///< 用处理图片宽和高表示的矩形
    int imageChannels;                      ///< 处理图片的通道数
	int imageType;                          ///< 处理图片的类型

	cv::Mat samples;                        ///< 保存先前像素值, 即样本
	std::vector<unsigned char*> rowSamples; ///< 样本的行首地址, 使用 vector 方便管理内存
    unsigned char** ptrSamples;             ///< &rowSamples[0], 使用数组的下标而不是 vector 的 [] 运算符, 加快程序运行速度

	cv::Mat noUpdateImage;                  ///< 不更新的像素, 如果对应位置上取非零值, 则该像素只识别是否为前景, 不更新模型
	std::vector<unsigned char*> rowNoUpdate;///< 行首地址, 使用 vector 方便管理内存
    unsigned char** ptrNoUpdate;            ///< &rowNoUpdate[0], 使用数组的下标而不是 vector 的 [] 运算符, 加快程序运行速度

private:
	void fill8UC3(const cv::Mat& image);
	void fill8UC1(const cv::Mat& image);
    void proc8UC3(const cv::Mat& image, cv::Mat& foreImage);
    void proc8UC1(const cv::Mat& image, cv::Mat& foreImage);

	int numOfSamples;                       ///< 每个像素保存的样本数量
	int minMatchDist;                       ///< 判定前景背景的距离
	int minNumOfMatchCount;                 ///< 判定为背景的最小匹配成功次数
	int subSampleInterval;                  ///< 它的倒数等于更新保存像素值的概率

	//! 在某一区间上均匀分布的整形随机数存储器
    class RandUniformInt
	{
	public:
        //! 初始化存储器
        /*!
            \param[in] size 随机数的数量
            \param[in] minInc 均匀分布的下界, 包含这个数
            \param[in] maxExc 均匀分布的上界, 不包含这个数
            \param[in] seed 随机数种子
         */
		void init(int size, int minInc, int maxExc, long long int seed)
		{
			capacity = size;
			minValInc = minInc;
			maxValExc = maxExc;
			mat.create(1, capacity, CV_32SC1);	
			cv::RNG rng(seed);
			rng.fill(mat, cv::RNG::UNIFORM, minValInc, maxValExc);
			data = (int*)mat.data;
			index = -1;	
		};
        //! 使用新的随机数种子 seed 更新存储的随机数
		void update(long long int seed)
		{
			cv::RNG rng(seed);
			rng.fill(mat, cv::RNG::UNIFORM, minValInc, maxValExc);
			index = -1;
		};
        //! 获取下一个随机数
		int getNext(void)
		{
			index++;
			if (index >= capacity || index < 0)
				index = 0;
			return data[index];
		}

	private:
		int minValInc;    ///< 均匀分布的下界, 包含这个值
        int maxValExc;    ///< 均匀分布的上界, 不包含这个值
		cv::Mat mat;      ///< 保存随机数的类
		int* data;        ///< mat 中第一个随机数的地址, 便于采用数组下标的方式访问保存在 mat 中的随机数
		int index;        ///< 被访问的随机数的下标
		int capacity;     ///< mat 中存储的随机数的数量
	};

	RandUniformInt rndReplaceCurr;          ///< 确定是否更新当前像素的样本
	RandUniformInt rndIndexCurr;            ///< 确定需替换的样本下标
	RandUniformInt rndReplaceAdj;           ///< 确定是否更新邻域像素样本
	RandUniformInt rndPositionAdj;          ///< 确定需要更新的邻域位置
	RandUniformInt rndIndexAdj;             ///< 确定续替换的样本的下标
};

//! 增加存储和获取背景图功能的 ViBe
class ExtendedViBe : public ViBe
{
public:
    struct ExtendedConfig : public Config
    {
        static ExtendedConfig getRGBConfig(void)
        {
            return ExtendedConfig(Config::getRGBConfig(), 0.02);
        }
        static ExtendedConfig getGrayConfig(void)
        {
            return ExtendedConfig(Config::getGrayConfig(), 0.02);
        }
        static ExtendedConfig getGradientConfig(void)
        {
            return ExtendedConfig(Config::getGradientConfig(), 0.02);
        }
        ExtendedConfig(const std::string& label_, 
            int numOfSamples_, int minMatchDist_, int minNumOfMatchCount_, int subSampleInterval_, float learnRate_)
            : Config(label_, 
            numOfSamples_, minMatchDist_, minNumOfMatchCount_, subSampleInterval_), learnRate(learnRate_)
        {}
        ExtendedConfig(const Config& config, float learnRate_)
            : Config(config), learnRate(learnRate_)
        {}
        float learnRate;
    };
	Z_LIB_EXPORT void init(const cv::Mat& image, const ExtendedConfig& config);
	Z_LIB_EXPORT void update(const cv::Mat& image, cv::Mat& foregroundImage, cv::Mat& backgroundImage, 
		const std::vector<cv::Rect>& rectsNoUpdate = std::vector<cv::Rect>());
	Z_LIB_EXPORT void refill(const cv::Mat& image);

private:
    cv::Mat backImage;     ///< 背景图
	float learnRate;       ///< 学习速率
	float compLearnRate;   ///< 1 - learnRate
};

}