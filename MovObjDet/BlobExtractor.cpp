#include <exception>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "BlobExtractor.h"
#include "ConfigFileReader.h"
#include "Segment.h"
#include "OperateData.h"
#include "ShowData.h"
#include "Exception.h"
#include "CompileControl.h"

namespace zsfo
{

class BlobExtractor::Impl
{
public:
    //! 初始化
	void init(const cv::Size& imageSize, const std::string& path, const std::string& label);
    // 修改配置参数
    void setConfigParams(
        const double* minObjectArea = 0, const double* minObjectWidth = 0, const double* minObjectHeight = 0, const bool* corrRatioCheck = 0, 
        const bool* charRegionCheck = 0, const std::vector<cv::Rect>& charRegionRects = std::vector<cv::Rect>(),
        const bool* merge = 0, const bool* mergeHori = 0, const bool* mergeVert = 0, const bool* mergeBigSmall = 0,
        const bool* refine = 0, const bool* refineByShape = 0, const bool* refineByGrad = 0, const bool* refineByColor = 0);
	//! 简单版本的处理函数
	void proc(cv::Mat& foreImage, const cv::Mat& image, const cv::Mat& backImage, 
        std::vector<cv::Rect>& rects, std::vector<cv::Rect>& stableRects);
    //! 处理函数
	void proc(cv::Mat& foreImage, const cv::Mat& image, const cv::Mat& backImage, const cv::Mat& gradDiffImage, 
        std::vector<cv::Rect>& rects, std::vector<cv::Rect>& stableRects);
    //! 在 image 上用颜色 color 画字符区域
	void drawCharRect(cv::Mat& image, const cv::Scalar& color);
	//! 在 image 上用颜色 color 画稳定的矩形区域
	void drawStableRects(cv::Mat& image, const cv::Scalar& color);
    //! 在 image 上用颜色 color 画最终矩形区域
    void drawFinalRects(cv::Mat& image, const cv::Scalar& color);

private:
	//! 物体结构体, 记录描述一个物体的矩形和轮廓, 白天模式使用
	struct Object
	{
        //! 物体的外接矩形
		cv::Rect rect;
        //! 物体的轮廓, 可能包括多个部分
		std::vector<std::vector<cv::Point> > contours;
	};
	//! 判断两组物体 objects1 和 objects2 是否完全相同, 并且排列顺序也完全相同
	bool areTheSameObjects(const std::vector<Object>& objects1, const std::vector<Object>& objects2);

	//! 前景矩形和相关信息，找位置稳定的矩形使用
	struct RectInfo
	{
		cv::Rect rect;     ///< 矩形
		int matchCount;    ///< 匹配的次数
		int missCount;     ///< 丢失的次数, 如果丢失的次数在一定范围内又匹配, 则清零

		RectInfo(void) : matchCount(0), missCount(0) {};
		RectInfo(const cv::Rect& rectVal) : rect(rectVal), matchCount(0), missCount(0) {}; 
	};

    // 形态学处理函数
    void morphOperation(cv::Mat& foreImage);
    // 找位置稳定的矩形
	void findStableRects(const std::vector<cv::Rect>& rects, std::vector<RectInfo>& rectInfos, std::vector<cv::Rect>& stableRects);

    // 简化版本函数
    // 优化物体函数
    void findRectsDayMode(cv::Mat& foreImage, const cv::Mat& image, const cv::Mat& backImage, std::vector<cv::Rect>& rects);
    // 合并物体函数
    void mergeRectsDayMode(const std::vector<cv::Rect>& srcRects, std::vector<cv::Rect>& dstRects);
    // 合并大小物体
    void mergeBigSmallRects(const std::vector<cv::Rect>& srcRects, std::vector<cv::Rect>& dstRects);
    // 合并水平方向上位置相近的物体
    void mergeHorizontalRects(const std::vector<cv::Rect>& srcRects, std::vector<cv::Rect>& dstRects);
    // 合并竖直方向上位置相近的物体
    void mergeVerticalRects(const std::vector<cv::Rect>& srcRects, std::vector<cv::Rect>& dstRects);

    // 白天模式找初始物体函数，过滤小的前景区域，进行路面筛查
    void findObjectsDayMode(cv::Mat& foreImage, const cv::Mat& normImage, const cv::Mat& backImage, std::vector<Object>& objects);
    // 优化物体函数，消除物体中的阴影区域和不属于真正物体的区域，进一步进行路面筛查
	void refineObjectsDayMode(const cv::Mat& normImage, const cv::Mat& backImage, const cv::Mat& gradDiffImage, 
		const std::vector<Object>& initObjects, std::vector<Object>& finalObjects);
	void refineObjectsDayModeNew(const cv::Mat& normImage, const cv::Mat& backImage, const cv::Mat& gradDiffImage, 
		const std::vector<Object>& initObjects, std::vector<Object>& finalObjects);
	// 合并物体函数
    void mergeObjectsDayMode(const std::vector<Object>& initObjects, std::vector<Object>& finalObjects);
    // 合并大小物体
    void mergeBigSmallObjects(const std::vector<Object>& srcObjects, std::vector<Object>& dstObjects);
    // 合并水平方向上位置相近的物体
    void mergeHorizontalObjects(const std::vector<Object>& srcObjects, std::vector<Object>& dstObjects);
    // 合并竖直方向上位置相近的物体
    void mergeVerticalObjects(const std::vector<Object>& srcObjects, std::vector<Object>& dstObjects);
	// 优化矩形函数
	cv::Rect refineRectByGradient(const cv::Mat& normImage, const cv::Mat& backImage, const cv::Mat& gradDiffImage, const cv::Mat& foreImage, const cv::Rect& currRect, 
		const cv::Mat& transNormImage, const cv::Mat& transBackImage, const cv::Mat& transGradDiffImage, const cv::Mat& transForeImage, const cv::Rect& transCurrRect);
	cv::Rect refineRectByShape(const cv::Mat& foreImage, const cv::Rect& currRect, const cv::Mat& transForeImage, const cv::Rect& transCurrRect);
	cv::Point findRectBoundsByGradient(const cv::Mat& normImage, const cv::Mat& backImage, 
		const cv::Mat& gradDiffImage, const cv::Mat& foreImage, const cv::Rect& currRect, bool vert);
	cv::Point findRectBoundsByShape(const cv::Mat& foreImage, const cv::Rect& currRect);
	// 根据前景区域找包围前景的最小矩形
	cv::Rect fitRectToForeground(const cv::Mat& foreImage, const cv::Rect& currRect, const cv::Mat& transForeImage, const cv::Rect& transCurrRect);
	// 优化具体某一个物体
	cv::Rect refineSingleObject(const cv::Mat& normImage, const cv::Mat& backImage, const cv::Mat& gradDiffImage, const cv::Mat& foreImage, const cv::Rect& currRect, 
		const cv::Mat& transNormImage, const cv::Mat& transBackImage, const cv::Mat& transGradDiffImage, const cv::Mat& transForeImage, const cv::Rect& transCurrRect);
	// 找新的边界
	void findPosStartAndEnd(const cv::Mat& foreImage, const cv::Rect& currRect, std::vector<int>& posStart, std::vector<int>& posEnd);
	cv::Point findRectOppositeBoundsByColorAndGrad(const std::vector<int>& posStart, const std::vector<int>& posEnd, 
		const cv::Mat& normImage, const cv::Mat& backImage, const cv::Mat& gradDiffImage, const cv::Rect& currRect);
	cv::Point findRectOppositeBoundsByShape(const std::vector<int>& posStart, const std::vector<int>& posEnd, const cv::Mat& normImage, const cv::Rect& currRect);

	// proc
	struct ConfigProc
	{
		bool runMerge;
		bool runRefine;
	};
	ConfigProc configProc;

	// findObjectsDayMode
	struct ConfigFODM
	{
		bool runWidthHeightCheck;
		bool runCharRegionCheck;
		bool runCorrRatioTest;
		double minObjectArea;        
		double minAvgWidth;
		double minAvgHeight;
        std::vector<cv::Rect> charRects;
		double minAreaRatioInCharRegion;
        double maxCorrRatioSum;
		double minHighCorrRatioB;
		double minHighCorrRatioG;
		double minHighCorrRatioR;
		double minLowCorrRatioB;
		double minLowCorrRatioG;
		double minLowCorrRatioR;
	};
	ConfigFODM configFODM;

	// mergeObjectsDayMode
	struct ConfigMODM
	{
		bool runMergeHori;
		bool runMergeVert;
		bool runMergeBigSmall;
	};
	ConfigMODM configMODM;

	// mergeHorizontalObjects
	struct ConfigMHO
	{
		double maxHeightRatioIntersectToCurr;
		double maxHeightRatioIntersectToTest;
		double minWidthRatioIntersectToUnion;
		double maxRatioWidthToHeight;
	};
	ConfigMHO configMHO;

	// mergeVerticalObjects
	struct ConfigMVO
	{
		double maxWidthRatioIntersectToCurr;
		double maxWidthRatioIntersectToTest;
		double minHeightRatioIntersectToUnion;
		double maxRatioHeightToWidth;
	};
	ConfigMVO configMVO;

	// mergeBigSmallObjects
	struct ConfigMBSO
	{
		double minAreaRatioTestToCurr;
		double minAreaRatioIntersectToBigTest;
		double minAreaRatioIntersectToSmallTest;
	};
	ConfigMBSO configMBSO;

	// refineObjectsDayMode
	struct ConfigRODM
	{
		bool runRefineByShape;
		bool runRefineByGradient;
		bool runFitRect;
	};
	ConfigRODM configRODM;

	// findRectBoundsByGradient
	struct ConfigFRBBG
	{
		bool runCheckByColor;
	};
	ConfigFRBBG configFRBBG;

	int imageWidth, imageHeight;
	cv::Rect fullBaseRect;

    std::vector<cv::Rect> rectsProc;
    std::vector<cv::Rect> rectsStable;
	std::vector<RectInfo> rectsRecords;
};

}

using namespace std;
using namespace cv;
using namespace ztool;

namespace zsfo
{

void BlobExtractor::init(const Size& imageSize, const string& path, const string& label)
{
	ptrImpl = new Impl;
	ptrImpl->init(imageSize, path, label);
}

void BlobExtractor::setConfigParams(
    const double* minObjectArea, const double* minObjectWidth, const double* minObjectHeight, const bool* corrRatioCheck, 
    const bool* charRegionCheck, const vector<Rect>& charRegionRects,
    const bool* merge, const bool* mergeHori, const bool* mergeVert, const bool* mergeBigSmall,
    const bool* refine, const bool* refineByShape, const bool* refineByGrad, const bool* refineByColor)
{
	ptrImpl->setConfigParams(minObjectArea, minObjectWidth, minObjectHeight, corrRatioCheck,
		charRegionCheck, charRegionRects, merge, mergeHori, mergeVert, mergeBigSmall,
		refine, refineByShape, refineByGrad, refineByColor);
}

void BlobExtractor::proc(Mat& foreImage, const Mat& image, const Mat& backImage, vector<Rect>& rects, vector<Rect>& stableRects)
{
	ptrImpl->proc(foreImage, image, backImage, rects, stableRects);
}

void BlobExtractor::proc(Mat& foreImage, const Mat& image, const Mat& backImage, const Mat& gradDiffImage, 
    vector<Rect>& rects, vector<Rect>& stableRects)
{
	ptrImpl->proc(foreImage, image, backImage, gradDiffImage, rects, stableRects);
}

void BlobExtractor::drawCharRect(Mat& image, const Scalar& color)
{
	ptrImpl->drawCharRect(image, color);
}

void BlobExtractor::drawStableRects(Mat& image, const Scalar& color)
{
	ptrImpl->drawStableRects(image, color);
}

void BlobExtractor::drawFinalRects(Mat& image, const Scalar& color)
{
	ptrImpl->drawFinalRects(image, color);
}

void BlobExtractor::Impl::init(const Size& imageSize, const string& path, const string& label)
{
    imageWidth = imageSize.width;
	imageHeight = imageSize.height;
	fullBaseRect = Rect(0, 0, imageWidth, imageHeight);

    if (!path.empty() && !label.empty())
    {
        ConfigFileReader reader(path);
        bool success;
        if (!reader.canBeOpened())
            //throw string("ERROR in BlobExtractor::init(), cannot open file ") + path;
            THROW_EXCEPT("cannot open file " + path);
        try
        {
            success = reader.read(label);                
        }
        catch (const exception& e)
        {
            //throw string("ERROR in BlobExtractor::init(), ") + e;
            THROW_EXCEPT(e.what());
        }
        if (!success)
            //throw string("ERROR in BlobExtractor::init(), cannot file label ") + label;
            THROW_EXCEPT("cannot file label " + label);

        reader.seek("(proc)");
        reader.getSingleKeySingleVal("#run_merge_objects_day_mode", configProc.runMerge);
        reader.getSingleKeySingleVal("#run_refine_objects_day_mode", configProc.runRefine);

        reader.seek("(find_objects_day_mode)");
        reader.getSingleKeySingleVal("#run_avg_width_height_check", configFODM.runWidthHeightCheck);
        reader.getSingleKeySingleVal("#run_char_region_check", configFODM.runCharRegionCheck);
        reader.getSingleKeySingleVal("#run_corr_ratio_check", configFODM.runCorrRatioTest);
        reader.getSingleKeySingleVal("#object_area", configFODM.minObjectArea);
        reader.getSingleKeySingleVal("#object_avg_width", configFODM.minAvgWidth);
        reader.getSingleKeySingleVal("#object_avg_height", configFODM.minAvgHeight);
        configFODM.charRects.clear();
        vector<vector<int> > ints;
        reader.getMultiKeyMultiVal("#char_region_rect_x_y_w_h", ints);
        if (!ints.empty())
        {
            int size = ints.size();            
            configFODM.charRects.reserve(size);
            for (int i = 0; i < size; i++)
            {
                if (ints[i].size() == 4)
                    configFODM.charRects.push_back(Rect(ints[i][0], ints[i][1], ints[i][2], ints[i][3]));
            }
        }
        reader.getSingleKeySingleVal("#ratio_object_in_char_region", configFODM.minAreaRatioInCharRegion);
        reader.getSingleKeySingleVal("#rect_fore_back_corr_ratio_sum", configFODM.maxCorrRatioSum);
        reader.getSingleKeySingleVal("#rect_fore_back_corr_ratio_b_high", configFODM.minHighCorrRatioB);
        reader.getSingleKeySingleVal("#rect_fore_back_corr_ratio_g_high", configFODM.minHighCorrRatioG);
        reader.getSingleKeySingleVal("#rect_fore_back_corr_ratio_r_high", configFODM.minHighCorrRatioR);
        reader.getSingleKeySingleVal("#rect_fore_back_corr_ratio_b_low", configFODM.minLowCorrRatioB);
        reader.getSingleKeySingleVal("#rect_fore_back_corr_ratio_g_low", configFODM.minLowCorrRatioG);
        reader.getSingleKeySingleVal("#rect_fore_back_corr_ratio_r_low", configFODM.minLowCorrRatioR);

        reader.seek("(merge_objects_day_mode)");
        reader.getSingleKeySingleVal("#run_merge_hori_objects", configMODM.runMergeHori);
        reader.getSingleKeySingleVal("#run_merge_vert_objects", configMODM.runMergeVert);
        reader.getSingleKeySingleVal("#run_merge_big_small_objects", configMODM.runMergeBigSmall);

        reader.seek("(merge_hori_objects)");
        reader.getSingleKeySingleVal("#height_ratio_intersect_to_curr", configMHO.maxHeightRatioIntersectToCurr);
        reader.getSingleKeySingleVal("#height_ratio_intersect_to_test", configMHO.maxHeightRatioIntersectToTest);
        reader.getSingleKeySingleVal("#width_ratio_intersect_to_union", configMHO.minWidthRatioIntersectToUnion);
        reader.getSingleKeySingleVal("#ratio_width_to_height", configMHO.maxRatioWidthToHeight);

        reader.seek("(merge_vert_objects)");
        reader.getSingleKeySingleVal("#width_ratio_intersect_to_curr", configMVO.maxWidthRatioIntersectToCurr);
        reader.getSingleKeySingleVal("#width_ratio_intersect_to_test", configMVO.maxWidthRatioIntersectToTest);
        reader.getSingleKeySingleVal("#height_ratio_intersect_to_union", configMVO.minHeightRatioIntersectToUnion);
        reader.getSingleKeySingleVal("#ratio_height_to_width", configMVO.maxRatioHeightToWidth);

        reader.seek("(merge_big_small_objects)");
        reader.getSingleKeySingleVal("#area_ratio_test_rect_to_curr_rect", configMBSO.minAreaRatioTestToCurr);
        reader.getSingleKeySingleVal("#area_ratio_intersect_rect_big_test_rect", configMBSO.minAreaRatioIntersectToBigTest);
        reader.getSingleKeySingleVal("#area_ratio_intersect_rect_small_test_rect", configMBSO.minAreaRatioIntersectToSmallTest);

        reader.seek("(refine_objects_day_mode)");
        reader.getSingleKeySingleVal("#run_refine_objects_by_shape", configRODM.runRefineByShape);
        reader.getSingleKeySingleVal("#run_refine_objects_by_gradient", configRODM.runRefineByGradient);
        reader.getSingleKeySingleVal("#run_fit_rect_to_foreground", configRODM.runFitRect);

        reader.seek("(find_rect_bounds_by_gradient)");
        reader.getSingleKeySingleVal("#run_check_by_color", configFRBBG.runCheckByColor);
    }
    else
    {
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("BlobExtractor is initialized with default param\n");
#endif
        configProc.runMerge = false;
        configProc.runRefine = false;

        configFODM.runWidthHeightCheck = true;
        configFODM.runCharRegionCheck = false;
        configFODM.runCorrRatioTest = true;
        configFODM.minObjectArea = 50;        
        configFODM.minAvgWidth = 5;
        configFODM.minAvgHeight = 5;
        configFODM.charRects.clear();
        configFODM.minAreaRatioInCharRegion = 0.5;
        configFODM.maxCorrRatioSum = 2.7;
        configFODM.minHighCorrRatioB = 0.85;
        configFODM.minHighCorrRatioG = 0.85;
        configFODM.minHighCorrRatioR = 0.85;
        configFODM.minLowCorrRatioB = 0.8;
        configFODM.minLowCorrRatioG = 0.8;
        configFODM.minLowCorrRatioR = 0.8;

        configMODM.runMergeHori = false;
        configMODM.runMergeVert = false;
        configMODM.runMergeBigSmall = false;

        configMHO.maxHeightRatioIntersectToCurr = 0.6;
        configMHO.maxHeightRatioIntersectToTest = 0.6;
        configMHO.minWidthRatioIntersectToUnion = -0.1;
        configMHO.maxRatioWidthToHeight = 2.5;

        configMVO.maxWidthRatioIntersectToCurr = 0.75;
        configMVO.maxWidthRatioIntersectToTest = 0.75;
        configMVO.minHeightRatioIntersectToUnion = -0.1;
        configMVO.maxRatioHeightToWidth = 1.75;

        configMBSO.minAreaRatioTestToCurr = 0.5;
        configMBSO.minAreaRatioIntersectToBigTest = 0.8;
        configMBSO.minAreaRatioIntersectToSmallTest = 0.7;

        configRODM.runRefineByShape = 0.5;
        configRODM.runRefineByGradient = 0.8;
        configRODM.runFitRect = 0.7;

        configFRBBG.runCheckByColor = true;
    }

#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
	printf("display blob extractor config:\n");

	printf("  function findBlobsDayMode:\n");
	printf("    configProc.runMerge = %s\n", configProc.runMerge ? "true" : "false");
	printf("    configProc.runRefine = %s\n", configProc.runRefine ? "true" : "false");

	printf("  function findObjectsDayMode:\n");
	printf("    configFODM.runWidthHeightCheck = %s\n", configFODM.runWidthHeightCheck ? "true" : "false");
	printf("    configFODM.runCharRegioncheck = %s\n", configFODM.runCharRegionCheck ? "true" : "false");
	printf("    configFODM.runCorrRatioTest = %s\n", configFODM.runCorrRatioTest ? "true" : "false");
	printf("    configFODM.minObjectArea = %.4f\n", configFODM.minObjectArea);	
	printf("    configFODM.minAvgWidth = %.4f\n", configFODM.minAvgWidth);
	printf("    configFODM.minAvgHeight = %.4f\n", configFODM.minAvgHeight);
    printf("    configFODM.charRects = ");
    if (!configFODM.charRects.empty())
    {
        int size = configFODM.charRects.size();
        for (int i = 0; i < size; i++)
        {
            printf("(x = %d, y = %d, w = %d, h = %d)", 
                configFODM.charRects[i].x, configFODM.charRects[i].y,
                configFODM.charRects[i].width, configFODM.charRects[i].height);
            if (i < size - 1)
                printf(", ");
        }
    }
    printf("\n");
    printf("    configFODM.minAreaRatioInCharRegion = %.4f\n", configFODM.minAreaRatioInCharRegion);
	printf("    configFODM.minHighCorrRatioB = %.4f\n", configFODM.minHighCorrRatioB);
	printf("    configFODM.minHighCorrRatioG = %.4f\n", configFODM.minHighCorrRatioG);
	printf("    configFODM.minHighCorrRatioR = %.4f\n", configFODM.minHighCorrRatioR);
	printf("    configFODM.minLowCorrRatioB = %.4f\n", configFODM.minLowCorrRatioB);
	printf("    configFODM.minLowCorrRatioG = %.4f\n", configFODM.minLowCorrRatioG);
	printf("    configFODM.minLowCorrRatioR = %.4f\n", configFODM.minLowCorrRatioR);

	printf("  function mergeObjectsDayMode:\n");
	printf("    configMODM.runMergeHori = %s\n", configMODM.runMergeHori ? "true" : "false");
	printf("    configMODM.runMergeVert = %s\n", configMODM.runMergeVert ? "true" : "false");
	printf("    configMODM.runMergeBigSmall = %s\n", configMODM.runMergeBigSmall ? "true" : "false");

	printf("  function mergeHorizontalObjects:\n");
	printf("    configMHO.maxHeightRatioIntersectToCurr = %.4f\n", configMHO.maxHeightRatioIntersectToCurr);
	printf("    configMHO.maxHeightRatioIntersectToTest = %.4f\n", configMHO.maxHeightRatioIntersectToTest);
	printf("    configMHO.minWidthRatioIntersectToUnion = %.4f\n", configMHO.minWidthRatioIntersectToUnion);
	printf("    configMHO.maxRatioWidthToHeight = %.4f\n", configMHO.maxRatioWidthToHeight);

	printf("  function mergeVerticalObjects:\n");
	printf("    configMVO.maxWidthRatioIntersectToCurr = %.4f\n", configMVO.maxWidthRatioIntersectToCurr);
	printf("    configMVO.maxWidthRatioIntersectToTest = %.4f\n", configMVO.maxWidthRatioIntersectToTest);
	printf("    configMVO.minHeightRatioIntersectToUnion = %.4f\n", configMVO.minHeightRatioIntersectToUnion);
	printf("    configMVO.maxRatioHeightToWidth = %.4f\n", configMVO.maxRatioHeightToWidth);

	printf("  function mergeBigSmallObjects:\n");
	printf("    configMBSO.minAreaRatioTestToCurr = %.4f\n", configMBSO.minAreaRatioTestToCurr);
	printf("    configMBSO.minAreaRatioIntersectToBigTest = %.4f\n", configMBSO.minAreaRatioIntersectToBigTest);
	printf("    configMBSO.minAreaRatioIntersectToSmallTest = %.4f\n", configMBSO.minAreaRatioIntersectToSmallTest);

	printf("  function refineObjectsDayMode:\n");
	printf("    configRODM.runRefineByShape = %s\n", configRODM.runRefineByShape ? "true" : "false");
	printf("    configRODM.runRefineByGradient = %s\n", configRODM.runRefineByGradient ? "true" : "false");
	printf("    configRODM.runFitRect = %s\n", configRODM.runFitRect ? "true" : "false");

	printf("  function findRectBoundsByGradient:\n");
	printf("    configFRBBG.runCheckByColor = %s\n", configFRBBG.runCheckByColor ? "true" : "false");
	printf("\n");
#endif
}

void BlobExtractor::Impl::setConfigParams(
    const double* minObjectArea, const double* minObjectWidth, const double* minObjectHeight, const bool* corrRatioCheck, 
    const bool* charRegionCheck, const vector<Rect>& charRegionRects,
    const bool* merge, const bool* mergeHori, const bool* mergeVert, const bool* mergeBigSmall,
    const bool* refine, const bool* refineByShape, const bool* refineByGrad, const bool* refineByColor)
{
    if (!(minObjectArea || minObjectWidth || minObjectHeight || corrRatioCheck || charRegionCheck ||
        merge || mergeHori || mergeVert || mergeBigSmall ||
        refine || refineByShape || refineByGrad || refineByColor))
        return;

#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
    printf("Some config param(s) of BlobExtractor set:\n");
#endif
    if (minObjectArea) 
    {
        configFODM.minObjectArea = *minObjectArea;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("  configFODM.minObjectArea = %.4f\n", configFODM.minObjectArea);
#endif
    }
    if (minObjectWidth)
    { 
        configFODM.minAvgWidth = *minObjectWidth;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("  configFODM.minAvgWidth = %.4f\n", configFODM.minAvgWidth);
#endif
    }
    if (minObjectHeight)
    {
        configFODM.minAvgHeight = *minObjectHeight;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("  configFODM.minAvgHeight = %.4f\n", configFODM.minAvgHeight);
#endif
    }
    if (corrRatioCheck)
    {
        configFODM.runCorrRatioTest = *corrRatioCheck;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("  configFODM.runCorrRatioTest = %s\n", configFODM.runCorrRatioTest ? "true" : "false");
#endif
    }
    if (charRegionCheck)
    {
        configFODM.runCharRegionCheck = *charRegionCheck;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("  configFODM.runCharRegionCheck = %s\n", configFODM.runCharRegionCheck ? "true" : "false");
#endif
        if (configFODM.runCharRegionCheck)
        {
            configFODM.charRects = charRegionRects;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
            printf("  configFODM.charRects = ");
            if (!configFODM.charRects.empty())
            {
                int size = configFODM.charRects.size();
                for (int i = 0; i < size; i++)
                {
                    printf("(x = %d, y = %d, w = %d, h = %d)", 
                        configFODM.charRects[i].x, configFODM.charRects[i].y,
                        configFODM.charRects[i].width, configFODM.charRects[i].height);
                    if (i < size - 1)
                        printf(", ");
                }
            }
            printf("\n");
#endif
        }
    }

    if (merge)
    {
        configProc.runMerge = *merge;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("  configProc.runMerge = %s\n", configProc.runMerge ? "true" : "false");
#endif
    }
    if (mergeHori)
    {
        configMODM.runMergeHori = *mergeHori;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("  configMODM.runMergeHori = %s\n", configMODM.runMergeHori ? "true" : "false");
#endif
    }
    if (mergeVert)
    { 
        configMODM.runMergeVert = *mergeVert;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("  configMODM.runMergeVert = %s\n", configMODM.runMergeVert ? "true" : "false");
#endif
    }
    if (mergeBigSmall)
    { 
        configMODM.runMergeBigSmall = *mergeBigSmall;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("  configMODM.runMergeBigSmall = %s\n", configMODM.runMergeBigSmall ? "true" : "false");
#endif
    }

    if (refine)
    { 
        configProc.runRefine = *refine;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("  configProc.runRefine = %s\n", configProc.runRefine ? "true" : "false");
#endif
    }
    if (refineByShape)
    { 
        configRODM.runRefineByShape = *refineByShape;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("  configRODM.runRefineByShape = %s\n", configRODM.runRefineByShape ? "true" : "false");
#endif
    }
    if (refineByGrad)
    { 
        configRODM.runRefineByGradient = *refineByGrad;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("  configRODM.runRefineByGradient = %s\n", configRODM.runRefineByGradient ? "true" : "false");
#endif
    }
    if (refineByColor)
    { 
        configFRBBG.runCheckByColor = *refineByColor;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("  configFRBBG.runCheckByColor = %s\n", configFRBBG.runCheckByColor ? "true" : "false");
#endif
    }
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
    printf("\n");
#endif
}

void BlobExtractor::Impl::morphOperation(Mat& foreImage)
{
    Mat coarseElement = getStructuringElement(MORPH_ELLIPSE, Size(7, 7), Point(-1, -1));
    Mat fineElement = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1));
    medianBlur(foreImage, foreImage, 3);
    dilate(foreImage, foreImage, coarseElement);
    erode(foreImage, foreImage, fineElement);
    //erode(foreImage, foreImage, fineElement);
    //medianBlur(foreImage, foreImage, 3);
    //dilate(foreImage, foreImage, coarseElement);
#if CMPL_SHOW_IMAGE
    imshow("foreground after morph operation", foreImage);
#endif
}

void BlobExtractor::Impl::findStableRects(const vector<Rect>& rects, vector<RectInfo>& rectInfos, vector<Rect>& stableRects)
{
    stableRects.clear();
    if (rects.empty()) return;

	// 如果 rectInfos 为空，则将当前帧找到的所有 rects 放到 rectInfos 中
	if (rectInfos.empty())
	{
		for (int i = 0; i < rects.size(); i++)
			rectInfos.push_back(RectInfo(rects[i]));
	}
	// 否则
	else
	{
		int numOfRects = rects.size();
		vector<RectInfo> currInfos(numOfRects);
		for (int i = 0; i < numOfRects; i++)
		{
			currInfos[i].rect = rects[i];
			currInfos[i].matchCount = 0;
		}
			
		// 如果 rectInfos 中的元素和某个 currInfos 中的元素能够完美匹配 则修改 rectInfos 中元素的 matchCount
		for (vector<RectInfo>::iterator itr = rectInfos.begin(); itr != rectInfos.end();)
		{
			bool match = false;
			for (int i = 0; i < numOfRects; i++)
			{
				if (currInfos[i].matchCount)
					continue;

				Rect intersectRect = itr->rect & currInfos[i].rect;
				Rect unionRect = itr->rect | currInfos[i].rect;
				if (unionRect.width > 20 && unionRect.height > 20 ? 
					intersectRect.area() > 0.95 * unionRect.area() : intersectRect.area() > 0.75 * unionRect.area())
				{
					currInfos[i].matchCount = 1;
					itr->rect = currInfos[i].rect;
					match = true;
					(itr->matchCount)++;
					if (itr->missCount > 0)
						itr->missCount = 0;
					break;
				}
			}

			if (!match)
			{
				(itr->missCount)++;
				if (itr->missCount > 15)
					itr = rectInfos.erase(itr);
				else
					++itr;
			}
			else
				++itr;
		}

		// 未能和 rectInfos 中任何元素匹配的 currInfos 中的元素直接放到 rectInfos 中
		for (int i = 0; i < numOfRects; i++)
		{
			if (!currInfos[i].matchCount)
				rectInfos.push_back(rects[i]);
		}

		// 检查 rectInfos 中所有元素 matchCount 的值
		// 大于阈值的矩形放到 stabelRects 中
		for (vector<RectInfo>::iterator itr = rectInfos.begin(); itr != rectInfos.end(); ++itr)
		{
			if (itr->matchCount > 20)
				stableRects.push_back(itr->rect);
		}
	}
}

void BlobExtractor::Impl::drawCharRect(Mat& image, const Scalar& color)
{
    if (configFODM.charRects.empty()) return;
    int size = configFODM.charRects.size();
    for (int i = 0; i < size; i++)
        rectangle(image, configFODM.charRects[i], color);
}

void BlobExtractor::Impl::drawStableRects(Mat& image, const Scalar& color)
{
	for (int i = 0; i < rectsStable.size(); i++)
	{
		rectangle(image, rectsStable[i], color);
	}
}

void BlobExtractor::Impl::drawFinalRects(Mat& image, const Scalar& color)
{
	for (int i = 0; i < rectsProc.size(); i++)
	{
		rectangle(image, rectsProc[i], color);
	}
}

void BlobExtractor::Impl::proc(Mat& foreImage, const Mat& image, const Mat& backImage, vector<Rect>& rects, vector<Rect>& stableRects)
{
	morphOperation(foreImage);
    if (!configProc.runMerge)
        findRectsDayMode(foreImage, image, backImage, rectsProc);
    else
    {
        vector<Rect> temp;
        findRectsDayMode(foreImage, image, backImage, temp);
        mergeRectsDayMode(temp, rectsProc);
    }
	findStableRects(rectsProc, rectsRecords, rectsStable);
	rects = rectsProc;
    stableRects = rectsStable;
#if CMPL_SHOW_IMAGE
    imshow("Processed Foreground Image", foreImage);
#endif
}

void BlobExtractor::Impl::findRectsDayMode(cv::Mat& foreImage, const Mat& image, const Mat& backImage, std::vector<cv::Rect>& rects)
{
    rects.clear();
    // 找轮廓
    vector< vector<Point> > initContours;
    findContours(foreImage, initContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // 如果轮廓的数量等于零，函数返回
    if (initContours.size() == 0)
        return;

    // 记录筛查后的轮廓
    vector<vector<Point> > contours;

    bool corrCheck = image.data && backImage.data && configFODM.runCorrRatioTest;

    // 对前景进行筛查
    for (int i = 0; i < initContours.size(); i++)
    {
        // 前景区域的面积
        double objectArea = contourArea(initContours[i]);
		if (objectArea < configFODM.minObjectArea)
            continue;

        // 获取外接矩形
        Rect currRect = boundingRect(initContours[i]);

		// 判断前景是否在字符区域中
		if (configFODM.runCharRegionCheck && !configFODM.charRects.empty())
		{
			int size = configFODM.charRects.size();
            int currRectArea = currRect.area();
            bool fallInCharRegion = false;
            for (int i = 0; i < size; i++)
            {
                Rect rectInCharRegion = currRect & configFODM.charRects[i];
			    if (rectInCharRegion.area() > /*0.5*/configFODM.minAreaRatioInCharRegion * currRectArea)
                {
                    fallInCharRegion = true;
                    break;
                }
            }
            if (fallInCharRegion)
                continue;
		}

        // 计算前景区域的平均宽度和平均高度
		if (configFODM.runWidthHeightCheck)
		{
			double avgObjectWidth = objectArea / currRect.height;
			double avgObjectHeight = objectArea / currRect.width;
			if (avgObjectWidth < configFODM.minAvgWidth ||
				avgObjectHeight < configFODM.minAvgHeight)
			{
#if CMPL_WRITE_CONSOLE
				printf("Current Rect: x = %d, y = %d, width = %d, height = %d\n",
						currRect.x, currRect.y, currRect.width, currRect.height);
				printf("Average object width is: %.2f, threshold is: %.2f\n",
						avgObjectWidth, configFODM.minAvgWidth);
				printf("Average object height is: %.2f, threshold is: %.2f\n",
						avgObjectHeight, configFODM.minAvgHeight);
				printf("Under qualification found, candidate object is filtered\n");
#endif
				continue;
			}
		}

        //相关系数检测
        if (corrCheck)
        {
            Scalar rectCorrRatio = calcCenterCorrRatio(image, backImage, currRect);
            double sumRatio = rectCorrRatio[0] + rectCorrRatio[1] + rectCorrRatio[2];
            if (sumRatio > configFODM.maxCorrRatioSum)
			{
#if CMPL_WRITE_CONSOLE
				printf("Current Rect: x = %d, y = %d, width = %d, height = %d\n",
						currRect.x, currRect.y, currRect.width, currRect.height);
				printf("Correlation ratios of between image rect and back rect are: %.4f %.4f %.4f\n",
						rectCorrRatio[0], rectCorrRatio[1], rectCorrRatio[2]);
				printf("Sum of them is: %.4f, threshold is: %.4f\n", sumRatio, configFODM.maxCorrRatioSum);
				printf("Sum of correlation ratios too large, candidate object is filtered\n");
#endif
				continue;
            }
        }
        rects.push_back(currRect);
        contours.push_back(initContours[i]);
    }

    // 在 foreImage 中画出最后的前景
    foreImage.setTo(0);
    drawContours(foreImage, contours, -1, Scalar(255), -1);
}

static bool areTheSameRects(const vector<Rect>& rects1, const vector<Rect>& rects2)
{
    if (rects1.size() != rects2.size())
        return false;

    bool retVal = true;
    for (int i = 0; i < rects1.size(); i++)
    {
        if (rects1[i] != rects2[i])
        {
            retVal = false;
            break;
        }
    }
    return retVal;
}

void BlobExtractor::Impl::mergeRectsDayMode(const vector<Rect>& srcRects, vector<Rect>& dstRects)
{
    dstRects.clear();
    if (srcRects.size() == 0)
        return;

	vector<Rect> src, dst;
	src = srcRects;
	if (configMODM.runMergeVert)
	{
		mergeVerticalRects(src, dst);
		src = dst;
	}
	if (configMODM.runMergeHori)
	{
		mergeHorizontalRects(src, dst);
		src = dst;
	}
	if (configMODM.runMergeBigSmall)
	{
		while (true)
		{
			mergeBigSmallRects(src, dst);
			if (areTheSameRects(src, dst))
				break;
			src = dst;
		}
	}
    dstRects.swap(dst);
}

void BlobExtractor::Impl::mergeBigSmallRects(const vector<Rect>& srcRects, vector<Rect>& dstRects)
{
    dstRects.clear();
	
	Rect currRect, testRect, unionRect, intersectRect;
    int rectsCount; // 记录矩形的个数
    bool* abandonMark; // 记录每个矩形是否被丢弃

    rectsCount = srcRects.size();
	vector<Rect> tempRects(srcRects);
    abandonMark = new bool [rectsCount];
    for (int i = 0; i < rectsCount; i++)
    {
        abandonMark[i] = false;
    }
    // 遍历所有矩形
    for (int i = 0; i < rectsCount; i++)
    {
        // 如果当前矩形已经被丢弃，则结束本次循环，进入下一个循环
        if (abandonMark[i] == true)
            continue;
        // 标记当前矩形，并将当前矩形 interRects[i] 赋予 currRect
        currRect = tempRects[i];
        // 遍历所有矩形
        for (int j = 0; j < rectsCount; j++)
        {
            if (i == j)
                continue;

            testRect = tempRects[j];
            // 只处理比 currRect 要小的矩形
            // 如果 testRect 比 currRect 要大，则结束本次循环
            if (testRect.width > currRect.width &&
                testRect.height > currRect.height)
                continue;
            // 计算矩形的交集
            intersectRect = testRect & currRect;
            // 处理 testRect 比较大的情况
			if (testRect.area() > currRect.area() * configMBSO.minAreaRatioTestToCurr/*0.5*/)
            {
                // 如果矩形的交集占 testRect 的面积较大，说明 testRect 很大一部分是和 currRect 重合的
                // 则丢弃 testRect，并且扩大 currRect
				if (intersectRect.area() > testRect.area() * configMBSO.minAreaRatioIntersectToBigTest/*0.8*/)
                {
                    abandonMark[j] = true;
                    currRect = currRect | testRect;
                }
                continue;
            }
            // 以下处理 testRect 比较小的情况
            // 如果交集矩形 intersectRect 和 testRect 面积相等，说明 testRect 完全在 currRect 内部
            if (intersectRect.area() == testRect.area())
            {
                abandonMark[j] = true;
                continue;
            }
            // 如果 testRect 只有一部分在 currRect 内部，则扩展 currRect
			if (intersectRect.area() > testRect.area() * configMBSO.minAreaRatioIntersectToSmallTest)
            {
                abandonMark[j] = true;
                currRect = currRect | testRect;
            }
        }
        tempRects[i] = currRect;
    }
	dstRects.reserve(rectsCount);
    for (int i = 0; i < rectsCount; i++)
    {
        if (abandonMark[i] == false)
		{
            dstRects.push_back(tempRects[i]);
		}
    }
    delete [] abandonMark;
}

void BlobExtractor::Impl::mergeHorizontalRects(const vector<Rect>& srcRects, vector<Rect>& dstRects)
{
    dstRects.clear();
	
	Rect currRect, testRect, unionRect, intersectRect;
    int intersectLeft, intersectRight, intersectTop, intersectBottom;
	int unionLeft, unionRight, unionTop, unionBottom;
    int intersectWidth, intersectHeight;
    int unionWidth, unionHeight;
    int rectsCount; // 记录矩形的个数
    bool* processMark; // 记录每个矩形是否已经被处理过

    rectsCount = srcRects.size();
    processMark = new bool [rectsCount];
    for (int i = 0; i < rectsCount; i++)
    {
        processMark[i] = false;
    }
    // 遍历所有矩形
    for (int i = 0; i < rectsCount; i++)
    {
        // 如果当前矩形已经被处理过，则结束本次循环，进入下一个循环
        if (processMark[i] == true)
            continue;
        // 标记当前矩形，并将当前矩形 interRects[i] 赋予 currRect
        processMark[i] = true;
        currRect = srcRects[i];
        // 遍历当前矩形后面的所有矩形
        for (int j = i + 1; j < rectsCount; j++)
        {
            // 如果当前矩形已经被处理过，则结束本次循环，进入下一个循环
            if (processMark[j] == true)
                continue;
            testRect = srcRects[j];
            // 检测 currRect 和 testRect 在竖直方向上的交集的长度和这两个矩形的高的比值
            // 如果两个比值中任意一者小于阈值，则这两个矩形不可能属于同一个前景
            intersectTop = max(currRect.y, testRect.y);
            intersectBottom = min(currRect.y + currRect.height, testRect.y + testRect.height);
            intersectHeight = intersectBottom - intersectTop;
			if (intersectHeight < /*0.6*/configMHO.maxHeightRatioIntersectToCurr * currRect.height ||
				intersectHeight < /*0.6*/configMHO.maxHeightRatioIntersectToTest * testRect.height)
                continue;
            // 计算 currRect 和 testRect 在横向上的并集和交集 unionWidth 和 intersectWidth
            // 如果 intersectWidth 和 unionWidth 的比值大于阈值，则这两个矩形不可能属于同一个前景
            unionLeft = min(currRect.x, testRect.x);
            unionRight = max(currRect.x + currRect.width, testRect.x + testRect.width);
            unionWidth = unionRight - unionLeft;
			intersectLeft = max(currRect.x, testRect.x);
			intersectRight = min(currRect.x + currRect.width, testRect.x + testRect.width);
            intersectWidth = intersectRight - intersectLeft;
			if (intersectWidth < /*-0.1*/configMHO.minWidthRatioIntersectToUnion * unionWidth)
                continue;
			// 计算 currRect 和 testRect 在竖直方向上的并集 unionHeight
			// 如果合并后的矩形在水平方向上太长 则这两个矩形不可能属于同一个前景
			unionTop = min(currRect.y, testRect.y);
			unionBottom = max(currRect.y + currRect.height, testRect.y + testRect.height);
			unionHeight = unionBottom - unionLeft;
			if (unionWidth > /*2.5*/configMHO.maxRatioWidthToHeight * unionHeight)
				continue;
            currRect = currRect | testRect;
            processMark[j] = true;
        }
        dstRects.push_back(currRect);
    }
    delete [] processMark;
}

void BlobExtractor::Impl::mergeVerticalRects(const vector<Rect>& srcRects, vector<Rect>& dstRects)
{
    dstRects.clear();
	
	Rect currRect, testRect, unionRect, intersectRect;
    int intersectLeft, intersectRight, intersectTop, intersectBottom;
	int unionLeft, unionRight, unionTop, unionBottom;
    int intersectWidth, intersectHeight;
    int unionWidth, unionHeight;
    int rectsCount; // 记录矩形的个数
    bool* processMark; // 记录每个矩形是否已经被处理过

    rectsCount = srcRects.size();
    processMark = new bool [rectsCount];
    for (int i = 0; i < rectsCount; i++)
    {
        processMark[i] = false;
    }
    // 遍历所有矩形
    for (int i = 0; i < rectsCount; i++)
    {
        // 如果当前矩形已经被处理过，则结束本次循环，进入下一个循环
        if (processMark[i] == true)
            continue;
        // 标记当前矩形，并将当前矩形 interRects[i] 赋予 currRect
        processMark[i] = true;
        currRect = srcRects[i];
        // 遍历当前矩形后面的所有矩形
        for (int j = i + 1; j < rectsCount; j++)
        {
            // 如果当前矩形已经被处理过，则结束本次循环，进入下一个循环
            if (processMark[j] == true)
                continue;
            testRect = srcRects[j];
            // 检测 currRect 和 testRect 在水平方向上的交集的长度和这两个矩形的宽的比值
            // 如果两个比值中任意一者小于阈值，则这两个矩形不可能属于同一个前景
            intersectLeft = max(currRect.x, testRect.x);
            intersectRight = min(currRect.x + currRect.width, testRect.x + testRect.width);
            intersectWidth = intersectRight - intersectLeft;
			if (intersectWidth < /*0.75*/configMVO.maxWidthRatioIntersectToCurr * currRect.width ||
				intersectWidth < /*0.75*/configMVO.maxWidthRatioIntersectToTest * testRect.width)
                continue;
            // 计算 currRect 和 testRect 在纵向上的并集和交集 unionHeight 和 intersectHeight
            // 如果 intersectHeight 和 unionHeight 的比值大于阈值，则这两个矩形不可能属于同一个前景
            unionTop = min(currRect.y, testRect.y);
            unionBottom = max(currRect.y + currRect.height, testRect.y + testRect.height);
            unionHeight = unionBottom - unionTop;
			intersectTop = max(currRect.y, testRect.y);
			intersectBottom = min(currRect.y + currRect.height, testRect.y + testRect.height);
            intersectHeight = intersectBottom - intersectTop;
			if (intersectHeight < /*-0.1*/configMVO.minHeightRatioIntersectToUnion * unionHeight)
                continue;
			// 计算 currRect 和 testRect 在水平方向上的并集 unionWidth
			// 如果合并后的矩形在竖直方向上太长 则这两个矩形不可能属于同一个前景
			unionLeft = min(currRect.x, testRect.x);
			unionRight = max(currRect.x + currRect.width, testRect.x + testRect.width);
			unionWidth = unionRight - unionLeft;
			if (unionHeight > /*1.75*/configMVO.maxRatioHeightToWidth * unionWidth)
				continue;
            currRect = currRect | testRect;
            processMark[j] = true;
        }
        dstRects.push_back(currRect);
    }
    delete [] processMark;
}

void BlobExtractor::Impl::proc(Mat& foreImage, const Mat& image, const Mat& backImage, const Mat& gradDiffImage, 
    vector<Rect>& rects, vector<Rect>& stableRects)
{
	morphOperation(foreImage);
	//findBlobsDayMode(image, foreImage, backImage, gradDiffImage, rectsProc);
    vector<Object> objects;
    bool runRefine = configProc.runRefine && gradDiffImage.data;
    if (!configProc.runMerge && !runRefine)
        findObjectsDayMode(foreImage, image, backImage, objects);
    else if (configProc.runMerge && !runRefine)
    {
        vector<Object> temp;
        findObjectsDayMode(foreImage, image, backImage, temp);
        mergeObjectsDayMode(temp, objects);
    }
    else
    {
        vector<Object> temp1, temp2;
        findObjectsDayMode(foreImage, image, backImage, temp1);
        mergeObjectsDayMode(temp1, temp2);
#if CMPL_USE_NEW_REFINE
		refineObjectsDayModeNew(image, backImage, gradDiffImage, temp2, objects);
#else
		refineObjectsDayMode(image, backImage, gradDiffImage, temp2, objects);
#endif
    }
    rectsProc.clear();
    int objectCount = objects.size();
    rectsProc.resize(objectCount);
    for (int i = 0; i < objectCount; i++)
        rectsProc[i] = objects[i].rect;
	findStableRects(rectsProc, rectsRecords, rectsStable);
    rects = rectsProc;
    stableRects = rectsStable;
#if CMPL_SHOW_IMAGE
    imshow("Processed Foreground Image", foreImage);
#endif
}

bool BlobExtractor::Impl::areTheSameObjects(const vector<Object>& objects1, const vector<Object>& objects2)
{
    if (objects1.size() != objects2.size())
        return false;

    bool retVal = true;
    for (int i = 0; i < objects1.size(); i++)
    {
        if (objects1[i].rect != objects2[i].rect)
        {
            retVal = false;
            break;
        }
    }
    return retVal;
}

void BlobExtractor::Impl::mergeObjectsDayMode(const vector<Object>& initObjects, vector<Object>& finalObjects)
{
    finalObjects.clear();
    if (initObjects.size() == 0)
        return;

	vector<Object> src, dst;
	src = initObjects;
	if (configMODM.runMergeVert)
	{
		mergeVerticalObjects(src, dst);
		src = dst;
	}
	if (configMODM.runMergeHori)
	{
		mergeHorizontalObjects(src, dst);
		src = dst;
	}
	if (configMODM.runMergeBigSmall)
	{
		while (true)
		{
			mergeBigSmallObjects(src, dst);
			if (areTheSameObjects(src, dst))
				break;
			src = dst;
		}
	}
    finalObjects.swap(dst);
}

void BlobExtractor::Impl::mergeBigSmallObjects(const vector<Object>& srcObjects, vector<Object>& dstObjects)
{
    dstObjects.clear();
	
	Rect currRect, testRect, unionRect, intersectRect;
    vector< vector<Point> > currContours, testContours;
    int objectsCount; // 记录矩形的个数
    bool* abandonMark; // 记录每个矩形是否被丢弃

    objectsCount = srcObjects.size();
	vector<Object> tempObjects(srcObjects);
    abandonMark = new bool [objectsCount];
    for (int i = 0; i < objectsCount; i++)
    {
        abandonMark[i] = false;
    }
    // 遍历所有矩形
    for (int i = 0; i < objectsCount; i++)
    {
        // 如果当前矩形已经被丢弃，则结束本次循环，进入下一个循环
        if (abandonMark[i] == true)
            continue;
        // 标记当前矩形，并将当前矩形 interRects[i] 赋予 currRect
        currRect = tempObjects[i].rect;
        currContours = tempObjects[i].contours;
        // 遍历所有矩形
        for (int j = 0; j < objectsCount; j++)
        {
            if (i == j)
                continue;

            testRect = tempObjects[j].rect;
            testContours = tempObjects[j].contours;
            // 只处理比 currRect 要小的矩形
            // 如果 testRect 比 currRect 要大，则结束本次循环
            if (testRect.width > currRect.width &&
                testRect.height > currRect.height)
                continue;
            // 计算矩形的交集
            intersectRect = testRect & currRect;
            // 处理 testRect 比较大的情况
			if (testRect.area() > currRect.area() * configMBSO.minAreaRatioTestToCurr/*0.5*/)
            {
                // 如果矩形的交集占 testRect 的面积较大，说明 testRect 很大一部分是和 currRect 重合的
                // 则丢弃 testRect，并且扩大 currRect
				if (intersectRect.area() > testRect.area() * configMBSO.minAreaRatioIntersectToBigTest/*0.8*/)
                {
                    abandonMark[j] = true;
                    currRect = currRect | testRect;
                    for (int k = 0; k < testContours.size(); k++)
                        currContours.push_back(testContours[k]);
                }
                continue;
            }
            // 以下处理 testRect 比较小的情况
            // 如果交集矩形 intersectRect 和 testRect 面积相等，说明 testRect 完全在 currRect 内部
            if (intersectRect.area() == testRect.area())
            {
                abandonMark[j] = true;
                continue;
            }
            // 如果 testRect 只有一部分在 currRect 内部，则扩展 currRect
			if (intersectRect.area() > testRect.area() * configMBSO.minAreaRatioIntersectToSmallTest)
            {
                abandonMark[j] = true;
                currRect = currRect | testRect;
                for (int k = 0; k < testContours.size(); k++)
                    currContours.push_back(testContours[k]);
            }
        }
        tempObjects[i].rect = currRect;
        tempObjects[i].contours = currContours;
    }
	dstObjects.reserve(objectsCount);
    for (int i = 0; i < objectsCount; i++)
    {
        if (abandonMark[i] == false)
		{
            dstObjects.push_back(Object());
			std::swap(tempObjects[i].rect, dstObjects.back().rect);
			tempObjects[i].contours.swap(dstObjects.back().contours);
		}
    }
    delete [] abandonMark;
}

void BlobExtractor::Impl::mergeHorizontalObjects(const vector<Object>& srcObjects, vector<Object>& dstObjects)
{
    dstObjects.clear();
	
	Rect currRect, testRect, unionRect, intersectRect;
    vector< vector<Point> > currContours, testContours;
    int intersectLeft, intersectRight, intersectTop, intersectBottom;
	int unionLeft, unionRight, unionTop, unionBottom;
    int intersectWidth, intersectHeight;
    int unionWidth, unionHeight;
    int objectsCount; // 记录矩形的个数
    bool* processMark; // 记录每个矩形是否已经被处理过

    objectsCount = srcObjects.size();
    processMark = new bool [objectsCount];
    for (int i = 0; i < objectsCount; i++)
    {
        processMark[i] = false;
    }
    // 遍历所有矩形
    for (int i = 0; i < objectsCount; i++)
    {
        // 如果当前矩形已经被处理过，则结束本次循环，进入下一个循环
        if (processMark[i] == true)
            continue;
        // 标记当前矩形，并将当前矩形 interRects[i] 赋予 currRect
        processMark[i] = true;
        currRect = srcObjects[i].rect;
        currContours = srcObjects[i].contours;
        // 遍历当前矩形后面的所有矩形
        for (int j = i + 1; j < objectsCount; j++)
        {
            // 如果当前矩形已经被处理过，则结束本次循环，进入下一个循环
            if (processMark[j] == true)
                continue;
            testRect = srcObjects[j].rect;
            testContours = srcObjects[j].contours;
            // 检测 currRect 和 testRect 在竖直方向上的交集的长度和这两个矩形的高的比值
            // 如果两个比值中任意一者小于阈值，则这两个矩形不可能属于同一个前景
            intersectTop = max(currRect.y, testRect.y);
            intersectBottom = min(currRect.y + currRect.height, testRect.y + testRect.height);
            intersectHeight = intersectBottom - intersectTop;
			if (intersectHeight < /*0.6*/configMHO.maxHeightRatioIntersectToCurr * currRect.height ||
				intersectHeight < /*0.6*/configMHO.maxHeightRatioIntersectToTest * testRect.height)
                continue;
            // 计算 currRect 和 testRect 在横向上的并集和交集 unionWidth 和 intersectWidth
            // 如果 intersectWidth 和 unionWidth 的比值大于阈值，则这两个矩形不可能属于同一个前景
            unionLeft = min(currRect.x, testRect.x);
            unionRight = max(currRect.x + currRect.width, testRect.x + testRect.width);
            unionWidth = unionRight - unionLeft;
			intersectLeft = max(currRect.x, testRect.x);
			intersectRight = min(currRect.x + currRect.width, testRect.x + testRect.width);
            intersectWidth = intersectRight - intersectLeft;
			if (intersectWidth < /*-0.1*/configMHO.minWidthRatioIntersectToUnion * unionWidth)
                continue;
			// 计算 currRect 和 testRect 在竖直方向上的并集 unionHeight
			// 如果合并后的矩形在水平方向上太长 则这两个矩形不可能属于同一个前景
			unionTop = min(currRect.y, testRect.y);
			unionBottom = max(currRect.y + currRect.height, testRect.y + testRect.height);
			unionHeight = unionBottom - unionLeft;
			if (unionWidth > /*2.5*/configMHO.maxRatioWidthToHeight * unionHeight)
				continue;
            currRect = currRect | testRect;
            for (int k = 0; k < testContours.size(); k++)
                currContours.push_back(testContours[k]);
            processMark[j] = true;
        }
        Object currObject;
        currObject.rect = currRect;
        currObject.contours = currContours;
        dstObjects.push_back(currObject);
    }
    delete [] processMark;
}

void BlobExtractor::Impl::mergeVerticalObjects(const vector<Object>& srcObjects, vector<Object>& dstObjects)
{
    dstObjects.clear();
	
	Rect currRect, testRect, unionRect, intersectRect;
    vector< vector<Point> > currContours, testContours;
    int intersectLeft, intersectRight, intersectTop, intersectBottom;
	int unionLeft, unionRight, unionTop, unionBottom;
    int intersectWidth, intersectHeight;
    int unionWidth, unionHeight;
    int objectsCount; // 记录矩形的个数
    bool* processMark; // 记录每个矩形是否已经被处理过

    objectsCount = srcObjects.size();
    processMark = new bool [objectsCount];
    for (int i = 0; i < objectsCount; i++)
    {
        processMark[i] = false;
    }
    // 遍历所有矩形
    for (int i = 0; i < objectsCount; i++)
    {
        // 如果当前矩形已经被处理过，则结束本次循环，进入下一个循环
        if (processMark[i] == true)
            continue;
        // 标记当前矩形，并将当前矩形 interRects[i] 赋予 currRect
        processMark[i] = true;
        currRect = srcObjects[i].rect;
        currContours = srcObjects[i].contours;
        // 遍历当前矩形后面的所有矩形
        for (int j = i + 1; j < objectsCount; j++)
        {
            // 如果当前矩形已经被处理过，则结束本次循环，进入下一个循环
            if (processMark[j] == true)
                continue;
            testRect = srcObjects[j].rect;
            testContours = srcObjects[j].contours;
            // 检测 currRect 和 testRect 在水平方向上的交集的长度和这两个矩形的宽的比值
            // 如果两个比值中任意一者小于阈值，则这两个矩形不可能属于同一个前景
            intersectLeft = max(currRect.x, testRect.x);
            intersectRight = min(currRect.x + currRect.width, testRect.x + testRect.width);
            intersectWidth = intersectRight - intersectLeft;
			if (intersectWidth < /*0.75*/configMVO.maxWidthRatioIntersectToCurr * currRect.width ||
				intersectWidth < /*0.75*/configMVO.maxWidthRatioIntersectToTest * testRect.width)
                continue;
            // 计算 currRect 和 testRect 在纵向上的并集和交集 unionHeight 和 intersectHeight
            // 如果 intersectHeight 和 unionHeight 的比值大于阈值，则这两个矩形不可能属于同一个前景
            unionTop = min(currRect.y, testRect.y);
            unionBottom = max(currRect.y + currRect.height, testRect.y + testRect.height);
            unionHeight = unionBottom - unionTop;
			intersectTop = max(currRect.y, testRect.y);
			intersectBottom = min(currRect.y + currRect.height, testRect.y + testRect.height);
            intersectHeight = intersectBottom - intersectTop;
			if (intersectHeight < /*-0.1*/configMVO.minHeightRatioIntersectToUnion * unionHeight)
                continue;
			// 计算 currRect 和 testRect 在水平方向上的并集 unionWidth
			// 如果合并后的矩形在竖直方向上太长 则这两个矩形不可能属于同一个前景
			unionLeft = min(currRect.x, testRect.x);
			unionRight = max(currRect.x + currRect.width, testRect.x + testRect.width);
			unionWidth = unionRight - unionLeft;
			if (unionHeight > /*1.75*/configMVO.maxRatioHeightToWidth * unionWidth)
				continue;
            currRect = currRect | testRect;
            for (int k = 0; k < testContours.size(); k++)
                currContours.push_back(testContours[k]);
            processMark[j] = true;
        }
        Object currObject;
        currObject.rect = currRect;
        currObject.contours = currContours;
        dstObjects.push_back(currObject);
    }
    delete [] processMark;
}

void BlobExtractor::Impl::findObjectsDayMode(Mat& foreImage, const Mat& normImage, const Mat& backImage, vector<Object>& objects)
{
    objects.clear();
    // 找轮廓
    vector< vector<Point> > initContours;
    findContours(foreImage, initContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // 如果轮廓的数量等于零，函数返回
    if (initContours.empty())
        return;

    // 记录筛查后的矩形和轮廓
    vector<Rect> rects;
    vector< vector<Point> > contours;

    bool corrCheck = normImage.data && backImage.data;

    // 对前景进行筛查
    for (int i = 0; i < initContours.size(); i++)
    {
        // 前景区域的面积
        double objectArea = contourArea(initContours[i]);
		if (objectArea < /*thresholds.objectArea*/configFODM.minObjectArea)
            continue;

        // 获取外接矩形
        Rect currRect = boundingRect(initContours[i]);

		// 判断前景是否在字符区域中
		if (configFODM.runCharRegionCheck && !configFODM.charRects.empty())
		{
			int size = configFODM.charRects.size();
            int currRectArea = currRect.area();
            bool fallInCharRegion = false;
            for (int i = 0; i < size; i++)
            {
                Rect rectInCharRegion = currRect & configFODM.charRects[i];
			    if (rectInCharRegion.area() > /*0.5*/configFODM.minAreaRatioInCharRegion * currRectArea)
                {
                    fallInCharRegion = true;
                    break;
                }
            }
            if (fallInCharRegion)
                continue;
		}

        // 计算前景区域的平均宽度和平均高度
		if (configFODM.runWidthHeightCheck)
		{
			double avgObjectWidth = objectArea / currRect.height;
			double avgObjectHeight = objectArea / currRect.width;
			if (avgObjectWidth < /*thresholds.objectAvgWidth*/configFODM.minAvgWidth ||
				avgObjectHeight < /*thresholds.objectAvgHeight*/configFODM.minAvgHeight)
			{
#if CMPL_WRITE_CONSOLE
				printf("Current Rect: x = %d, y = %d, width = %d, height = %d\n",
						currRect.x, currRect.y, currRect.width, currRect.height);
				printf("Average object width is: %.2f, threshold is: %.2f\n",
						avgObjectWidth, configFODM.minAvgWidth);
				printf("Average object height is: %.2f, threshold is: %.2f\n",
						avgObjectHeight, configFODM.minAvgHeight);
				printf("Under qualification found, candidate object is filtered\n");
#endif
				continue;
			}
		}

        // 计算当前矩形区域内彩色图和背景图的相关系数
        if (configFODM.runCorrRatioTest && corrCheck)
		{
			Scalar rectCorrRatio = calcCenterCorrRatio(normImage, backImage, currRect);
			if (rectCorrRatio[0] > configFODM.minHighCorrRatioB && rectCorrRatio[1] > configFODM.minHighCorrRatioG ||
				rectCorrRatio[0] > configFODM.minHighCorrRatioB && rectCorrRatio[2] > configFODM.minHighCorrRatioR ||
				rectCorrRatio[1] > configFODM.minHighCorrRatioG && rectCorrRatio[2] > configFODM.minHighCorrRatioR)
			{
#if CMPL_WRITE_CONSOLE
				printf("Current Rect: x = %d, y = %d, width = %d, height = %d\n",
						currRect.x, currRect.y, currRect.width, currRect.height);
				printf("Correlation ratios of between image rect and back rect are: %.4f %.4f %.4f\n",
						rectCorrRatio[0], rectCorrRatio[1], rectCorrRatio[2]);
				printf("Thresholds are: %.4f %.4f %.4f\n", 
					configFODM.minHighCorrRatioB, configFODM.minHighCorrRatioG, configFODM.minHighCorrRatioR);
				printf("At least two of the correlation ratios too large, candidate object is filtered\n");
#endif
				continue;
			}				
			if (rectCorrRatio[0] > configFODM.minLowCorrRatioB &&
				rectCorrRatio[1] > configFODM.minLowCorrRatioG &&
				rectCorrRatio[2] > configFODM.minLowCorrRatioR)
			{
#if CMPL_WRITE_CONSOLE	
				printf("Current Rect: x = %d, y = %d, width = %d, height = %d\n",
						currRect.x, currRect.y, currRect.width, currRect.height);
				printf("Correlation ratios of between image rect and back rect are: %.4f %.4f %.4f\n",
						rectCorrRatio[0], rectCorrRatio[1], rectCorrRatio[2]);
				printf("Thresholds are: %.4f %.4f %.4f\n", 
					configFODM.minLowCorrRatioB, configFODM.minLowCorrRatioG, configFODM.minLowCorrRatioR);
				printf("All of the correlation ratios too large, candidate object is filtered\n");
#endif
				continue;
			}
		}

        // 经过上述筛查的前景被认为是真正的前景
        /*
#if CMPL_WRITE_CONSOLE	
        printf("Current Rect: x = %d, y = %d, width = %d, height = %d\n",
                currRect.x, currRect.y, currRect.width, currRect.height);
        printf("Undergone all checks, object qualified.\n");
#endif */
        rects.push_back(currRect);
        contours.push_back(initContours[i]);
    }

    // 将查找结果 push 到 objects 向量中
    for (int i = 0; i < rects.size(); i++)
    {
        Object currObject;		
        currObject.rect = rects[i];
        currObject.contours.push_back(contours[i]);
		// 找轮廓的凸集，采用下面的代码
		/*vector<int> indices;
		convexHull(contours[i], indices);
		vector<Point> points(indices.size());
		for (int j = 0; j < indices.size(); j++)
		{
			points[j] = contours[i][indices[j]];
		}
		currObject.contours.push_back(points);
		contours[i] = points;*/
		objects.push_back(currObject);
    }
    // 在 foreImage 中画出最后的前景
    foreImage.setTo(0);
    drawContours(foreImage, contours, -1, Scalar(255), -1);
}

void BlobExtractor::Impl::refineObjectsDayMode(const Mat& normImage, const Mat& backImage, const Mat& gradDiffImage, 
	const vector<Object>& initObjects, vector<Object>& finalObjects)
{
    finalObjects.clear();
    if (initObjects.empty())
        return;

    Mat gradImage = gradDiffImage;
    Mat currObjectImage = Mat::zeros(gradDiffImage.rows, gradDiffImage.cols, CV_8UC1);
    Mat transCurrObjectImage = Mat::zeros(gradDiffImage.cols, gradDiffImage.rows, CV_8UC1);
    Mat transGradImage = gradImage.t();
	Mat transNormImage = normImage.t();
	Mat transBackImage = backImage.t();
    for (int i = 0; i < initObjects.size(); i++)
    {
        currObjectImage.setTo(0);
        drawContours(currObjectImage, initObjects[i].contours, -1, Scalar(255), -1);
        transCurrObjectImage = currObjectImage.t();
        Rect currRect = initObjects[i].rect;
        Rect transCurrRect = Rect(currRect.y, currRect.x, currRect.height, currRect.width);
#if CMPL_WRITE_CONSOLE
        printf("Current Rect before refine:            x = %3d, y = %3d, width = %3d, height = %3d\n",
                currRect.x, currRect.y, currRect.width, currRect.height);
#endif
        // 根据形状信息优化矩形
		if (configRODM.runRefineByShape)
		{
			currRect = refineRectByShape(currObjectImage, currRect, transCurrObjectImage, transCurrRect);
			transCurrRect = Rect(currRect.y, currRect.x, currRect.height, currRect.width);
#if CMPL_WRITE_CONSOLE
			printf("Current Rect after refine by shape:    x = %3d, y = %3d, width = %3d, height = %3d\n",
					currRect.x, currRect.y, currRect.width, currRect.height);
#endif
		}
        // 根据梯度信息优化矩形
		if (configRODM.runRefineByGradient)
		{
			currRect = refineRectByGradient(normImage, backImage, gradImage, currObjectImage, currRect,
 				transNormImage, transBackImage, transGradImage, transCurrObjectImage, transCurrRect);
			transCurrRect = Rect(currRect.y, currRect.x, currRect.height, currRect.width);
#if CMPL_WRITE_CONSOLE
			printf("Current Rect after refine by gradient: x = %3d, y = %3d, width = %3d, height = %3d\n",
					currRect.x, currRect.y, currRect.width, currRect.height);
#endif
			if (currRect.width == 0 && currRect.height == 0)
			{
#if CMPL_WRITE_CONSOLE
				printf("Very similar to the back image, candidate object is filtered\n");
#endif
				continue;
			}
		}
		// 根据前景调整矩形
		if (configRODM.runFitRect)
			currRect = fitRectToForeground(currObjectImage, currRect, transCurrObjectImage, transCurrRect);

        Object currObject;
        currObject.rect = currRect & Rect(0, 0, normImage.cols, normImage.rows);
        currObject.contours = initObjects[i].contours;
        finalObjects.push_back(currObject);
    }
}

Rect BlobExtractor::Impl::refineRectByGradient(const Mat& normImage, const Mat& backImage, const Mat& gradDiffImage, const Mat& foreImage, const Rect& currRect, 
	const Mat& transNormImage, const Mat& transBackImage, const Mat& transGradDiffImage, const Mat& transForeImage, const Rect& transCurrRect)
{
	// 先找左右边界
	Point horiBound = findRectBoundsByGradient(transNormImage, transBackImage, 
		transGradDiffImage, transForeImage, transCurrRect, false);
    int left = horiBound.x;
    int right = horiBound.y;

#if CMPL_WRITE_CONSOLE
    if (left != 0 || right != currRect.width - 1)
    {
        printf("Current Rect: x = %d, y = %d, width = %d, height = %d Changed in findRectBoundariesByGradient\n",
               currRect.x, currRect.y, currRect.width, currRect.height);
        printf("Orig Width: %3d. New hori boundaries: left: %d, right: %d\n", currRect.width, left, right);
    }
#endif
	// 如果左右边界都很极端，则当前目标不是真实运动物体
	if (right - left < currRect.width * 0.15 ||
		left > 0.85 * currRect.width && right < 0.15 * currRect.width)
    {
        return Rect(0, 0, 0, 0);
    }
	// 微调边界
	if (left < 0.075 * currRect.width || left > 0.8 * currRect.width)
        left = 0;
    if (right > 0.925 * currRect.width || right < 0.2 * currRect.width)
        right = currRect.width;
    else
        right++;

	Rect interRect = Rect(currRect.x + left, currRect.y, right - left, currRect.height);
	// 找上下边界
	Point vertBound = findRectBoundsByGradient(normImage, backImage, gradDiffImage, foreImage, interRect, true);
	int top = vertBound.x;
    int bottom = vertBound.y;

#if CMPL_WRITE_CONSOLE
	if (top != 0 || bottom != currRect.height - 1)
    {
        printf("Current Rect: x = %d, y = %d, width = %d, height = %d Changed in findRectBoundariesByGradient\n",
               currRect.x, currRect.y, currRect.width, currRect.height);
        printf("Orig Height: %3d. New vert boundaries: top: %d, bottom: %d\n", currRect.height, top, bottom);
    }
#endif
	// 如果左右边界都很极端，则当前目标不是真实运动物体
    if (bottom - top < currRect.height * 0.15 ||
        top > 0.85 * currRect.height && bottom < 0.15 * currRect.height)
    {
        return Rect(0, 0, 0, 0);
    }	
	top = 0;
	// 微调边界
    if (top < 0.075 * currRect.height || top > 0.65 * currRect.height)
        top = 0;
    if (bottom > 0.925 * currRect.height || bottom < 0.35 * currRect.height)
        bottom = currRect.height;
    else
        bottom += 5;

	return Rect(currRect.x + left, currRect.y + top, right - left, bottom - top) &
           Rect(0, 0, normImage.cols, normImage.rows);
}

Rect BlobExtractor::Impl::refineRectByShape(const Mat& foreImage, const Rect& currRect, const Mat& transForeImage, const Rect& transCurrRect)
{
	Point vertBound = findRectBoundsByShape(foreImage, currRect);
	Point horiBound = findRectBoundsByShape(transForeImage, transCurrRect);

	int top = vertBound.x;
	int bottom = vertBound.y;
	int left = horiBound.x;
	int right = horiBound.y;

	if (top < 10 || top < 0.05 * currRect.height)
		top = 0;
	if (bottom + 10 > currRect.height || bottom > 0.95 * currRect.height)
		bottom = currRect.height;
	else
		bottom++;
	if (left < 10 || left < 0.05 * currRect.width)
		left = 0;
	if (right + 10 > currRect.width || right > 0.95 * currRect.width)
		right = currRect.width;
	else
		right++;

	Rect retRect;
	retRect.x = currRect.x + left;
    retRect.y = currRect.y + top;
    retRect.width = right - left;
    retRect.height = bottom - top;

	return retRect;
}

Point BlobExtractor::Impl::findRectBoundsByGradient(const Mat& normImage, const Mat& backImage, 
	const Mat& gradDiffImage, const Mat& foreImage, const Rect& currRect, bool findVertBounds)
{
    //int* horiStart = new int [currRect.height];
    //int* horiEnd = new int [currRect.height];
    int initHoriStart = currRect.width;
    int initHoriEnd = -1;
    /*for (int i = 0; i < currRect.height; i++)
    {
        horiStart[i] = initHoriStart;
        horiEnd[i] = initHoriEnd;
    }*/
	vector<int> horiStart(currRect.height, initHoriStart);
	vector<int> horiEnd(currRect.height, initHoriEnd);

    // 查找每一行的起始编号和终止编号
    for (int i = 0; i < currRect.height; i++)
    {
        const unsigned char* ptrMark = foreImage.ptr<unsigned char>(currRect.y + i) + currRect.x;
        //float* ptrMark = gradDiffImage.ptr<float>(currRect.y + i) + currRect.x;
        for (int j = 0; j < currRect.width; j++)
        {
            if (ptrMark[j] > 0)
            {
                horiStart[i] = j;
                break;
            }
        }
    }
    for (int i = 0; i < currRect.height; i++)
    {
        const unsigned char* ptrMark = foreImage.ptr<unsigned char>(currRect.y + i) + currRect.x;
        //float* ptrMark = gradDiffImage.ptr<float>(currRect.y + i) + currRect.x;
        for (int j = currRect.width - 1; j >= 0; j--)
        {
            if (ptrMark[j] > 0)
            {
                horiEnd[i] = j;
                break;
            }
        }
    }

    // 标记每一行是否为候选阴影行
    //unsigned char* horiIsShadow = new unsigned char [currRect.height];
	vector<unsigned char> horiIsShadow(currRect.height);
    for (int i = 0; i < currRect.height; i++)
    {
        if (horiStart[i] == initHoriStart && horiEnd[i] == initHoriEnd/* ||
            horiEnd[i] - horiStart[i] < 0.1 * currRect.width*/)
        {
            horiIsShadow[i] = 1;
            continue;
        }

		int length = horiEnd[i] - horiStart[i] + 1;
        horiIsShadow[i] = 0;

		vector<Segment<unsigned char> > horiSegments;
        const unsigned char* ptrMark = gradDiffImage.ptr<unsigned char>(currRect.y + i) + currRect.x;
        findSegments(ptrMark + horiStart[i], length, horiSegments);
		/*if (horiSegments.size() == 1 && horiSegments[0].data == 0)
		{
			horiIsShadow[i] = 1;
			continue;
		}*/

		bool isShadByGradDiff = false;		
        for (int j = 0; j < horiSegments.size(); j++)
        {
            if (horiSegments[j].data == 0.0F &&
				(horiSegments[j].length > 0.6 * length || 
				 horiSegments.size() > j + 2 && horiSegments[j + 1].length < 15 && 
				 horiSegments[j].length + horiSegments[j + 1].length + horiSegments[j + 2].length > 0.7 * length))
			{
				isShadByGradDiff = true;
				break;
            }
        }

        if (!configFRBBG.runCheckByColor || !normImage.data || !backImage.data)
		{
			horiIsShadow[i] = isShadByGradDiff;
			continue;
		}

		int lineShadCount = 0;
		const unsigned char* ptrNormData = normImage.ptr<unsigned char>(currRect.y + i) + currRect.x * 3;
		const unsigned char* ptrBackData = backImage.ptr<unsigned char>(currRect.y + i) + currRect.x * 3;
		for (int j = horiStart[i]; j <= horiEnd[i]; j++)
		{
			if (ptrNormData[j * 3] < ptrBackData[j * 3] &&
				ptrNormData[j * 3 + 1] < ptrBackData[j * 3 + 1] &&
				ptrNormData[j * 3 + 2] < ptrBackData[j * 3 + 2])
			{
				unsigned char maxVal;
				maxVal = ptrNormData[j * 3] > ptrNormData[j * 3 + 1] ? ptrNormData[j * 3] : ptrNormData[j * 3 + 1];
				maxVal = ptrNormData[j * 3 + 2] > maxVal ? ptrNormData[j * 3 + 2] : maxVal;
				if (maxVal == 0)
				{
					lineShadCount++;
					continue;
				}
				unsigned char minVal;
				minVal = ptrNormData[j * 3] < ptrNormData[j * 3 + 1] ? ptrNormData[j * 3] : ptrNormData[j * 3 + 1];
				minVal = ptrNormData[j * 3 + 2] < minVal ? ptrNormData[j * 3 + 2] : minVal;
				if ((maxVal - minVal) < 0.3 * maxVal)
					lineShadCount++;
			}
		}
		bool isShadByColor = lineShadCount > length * 0.7;        
		horiIsShadow[i] = isShadByColor && isShadByGradDiff;
    }
#if CMPL_SHOW_IMAGE
	if (!findVertBounds)
		showArrayByVertBar("is col shadow", horiIsShadow, false, true);
	if (findVertBounds)
		showArrayByHoriBar("is row shadow", horiIsShadow, false, true);
#endif

    // 给判决结果数组 horiIsShadow 分段
    vector<Segment<unsigned char> > horiJudgeSegments;
    findSegments(horiIsShadow, horiJudgeSegments);
	int top = 0, bottom = currRect.height - 1;
    // 找新的竖直方向上的边界点
	for (int i = 0; i < horiJudgeSegments.size(); i++)
	{
		if (horiJudgeSegments[i].begin < int(currRect.height * 0.1) &&
			horiJudgeSegments[i].length > int(currRect.height * 0.075) &&
			horiJudgeSegments[i].data ||
			horiJudgeSegments[i].begin < int(currRect.height * 0.3) &&
			horiJudgeSegments[i].length > int(currRect.height * 0.2) &&
			horiJudgeSegments[i].data)
		{
			top = horiJudgeSegments[i].end;
			break;
		}
	}
	for (int i = horiJudgeSegments.size() - 1; i >= 0; i--)
	{
		if (horiJudgeSegments[i].end > int(currRect.height * 0.9) &&
			horiJudgeSegments[i].length > int(currRect.height * 0.075) &&
			horiJudgeSegments[i].data ||
			horiJudgeSegments[i].end > int(currRect.height * 0.7) &&
			horiJudgeSegments[i].length > int(currRect.height * 0.2) &&
			horiJudgeSegments[i].data)
		{
			bottom = horiJudgeSegments[i].begin;
			break;
		}
	}
	if (!findVertBounds)
	{
		// 找长度最大的段
		int maxIndex = -1;
		int maxLength = 0;
		for (int i = 0; i < horiJudgeSegments.size(); i++)
		{
			if (horiJudgeSegments[i].data && horiJudgeSegments[i].length > maxLength)
			{
				maxIndex = i;
				maxLength = horiJudgeSegments[i].length;
			}
		}
		if (maxIndex >= 0)
		{
			// 计算这个段上方和下方剩余的部分长度的均值
			int avgWidthTop = 0, avgWidthBottom = 0;
			int heightTop = 0, heightBottom = 0;
			if (horiJudgeSegments[maxIndex].begin > 0)
			{
				heightTop = horiJudgeSegments[maxIndex].begin;
				for (int i = 0; i < horiJudgeSegments[maxIndex].begin; i++)
					avgWidthTop += currRect.width - horiStart[i];
				avgWidthTop = double(avgWidthTop) / heightTop;
			}
			if (horiJudgeSegments[maxIndex].end < currRect.height - 1)
			{
				heightBottom = currRect.height - horiJudgeSegments[maxIndex].end - 1;
				for (int i = horiJudgeSegments[maxIndex].end + 1; i < currRect.height; i++)
					avgWidthBottom += currRect.width - horiStart[i];
				avgWidthBottom = double(avgWidthBottom) / heightBottom;
			}
			// 如果找到的最大长度段比较长
			if (maxLength > 0.2 * currRect.height)
			{
				if (avgWidthTop > avgWidthBottom &&
					avgWidthTop - avgWidthBottom > 0.2 * currRect.width &&
					heightTop > currRect.height * 0.2)
				{
					bottom = horiJudgeSegments[maxIndex].begin;
					if (top > bottom)
						top = 0;
				}
				if (avgWidthTop < avgWidthBottom &&
					avgWidthBottom - avgWidthTop > 0.2 * currRect.width &&
					heightBottom > currRect.height * 0.2)
				{
					top = horiJudgeSegments[maxIndex].end;
					if (bottom < top)
						bottom = currRect.height - 1;
				}
			}
		}
	}

    /*delete [] horiIsShadow;
    delete [] horiStart;
    delete [] horiEnd;*/

	if (top >= bottom ||
		top < bottom && bottom - top < currRect.height * 0.2)
		return Point(0, currRect.height - 1);
	else
		return Point(top, bottom);
}

Point BlobExtractor::Impl::findRectBoundsByShape(const Mat& foreImage, const Rect& currRect)
{
	if (currRect.width < 50)
		return Point(0, currRect.height - 1);
	
	//int* horiStart = new int [currRect.height];
    //int* horiEnd = new int [currRect.height];
    int initHoriStart = currRect.width;
    int initHoriEnd = -1;
    /*for (int i = 0; i < currRect.height; i++)
    {
        horiStart[i] = initHoriStart;
        horiEnd[i] = initHoriEnd;
    }*/
	vector<int> horiStart(currRect.height, initHoriStart);
	vector<int> horiEnd(currRect.height, initHoriEnd);

    // 查找每一行的起始编号和终止编号
    for (int i = 0; i < currRect.height; i++)
    {
        const unsigned char* ptrMark = foreImage.ptr<unsigned char>(currRect.y + i) + currRect.x;
        for (int j = 0; j < currRect.width; j++)
        {
            if (ptrMark[j] > 0)
            {
                horiStart[i] = j;
                break;
            }
        }
    }
    for (int i = 0; i < currRect.height; i++)
    {
        const unsigned char* ptrMark = foreImage.ptr<unsigned char>(currRect.y + i) + currRect.x;
        for (int j = currRect.width - 1; j >= 0; j--)
        {
            if (ptrMark[j] > 0)
            {
                horiEnd[i] = j;
                break;
            }
        }
    }

	// 判断每一行是宽还是窄
	vector<unsigned char> isRowNarrow(currRect.height, 0);
    for (int i = 0; i < currRect.height; i++)
    {
        if (horiStart[i] == initHoriStart && horiEnd[i] == initHoriEnd)
            continue;
        vector<Segment<unsigned char> > horiSegments;
        const unsigned char* ptrMark = foreImage.ptr<unsigned char>(currRect.y + i) + currRect.x;
        int length = horiEnd[i] - horiStart[i] + 1;
        findSegments(ptrMark + horiStart[i], length, horiSegments);
		if (horiSegments.size() == 1 && length < 20)
			isRowNarrow[i] = 1;

    }
	vector<Segment<unsigned char> > horiJudgeSegments;
	findSegments(isRowNarrow, horiJudgeSegments);

	// 去除前景中的细长条区域
	int top = 0, bottom = currRect.height - 1;
	for (int i = 0; i < horiJudgeSegments.size() - 1; i++)
	{
		if (horiJudgeSegments[i].begin < currRect.height / 2 &&
			horiJudgeSegments[i].data && 
			horiJudgeSegments[i].length > currRect.height * 0.1)
			top = horiJudgeSegments[i].end;
	}
	for (int i = horiJudgeSegments.size() - 1; i >= 0; i--)
	{
		if (horiJudgeSegments[i].end > currRect.height / 2 &&
			horiJudgeSegments[i].data && 
			horiJudgeSegments[i].length > currRect.height * 0.1)
			bottom = horiJudgeSegments[i].begin;
	}

	/*// 去除前景中的细长条区域
	int top = 0, bottom = currRect.height - 1;
    // 找起始行
    for (int i = 0; i < currRect.height; i++)
    {
        if (horiStart[i] == initHoriStart && horiEnd[i] == initHoriEnd)
            continue;
        vector<Segment<unsigned char> > horiSegments;
        unsigned char* ptrMark = foreImage.ptr<unsigned char>(currRect.y + i) + currRect.x;
        int length = horiEnd[i] - horiStart[i] + 1;
        findSegments(ptrMark + horiStart[i], length, horiSegments);
		if (horiSegments.size() == 1 && length > 20)
		{
			top = i;
			break;
		}
    }
	// 找结束行
    for (int i = currRect.height - 1; i >= 0; i--)
    {
        if (horiStart[i] == initHoriStart && horiEnd[i] == initHoriEnd)
            continue;
        vector<Segment<unsigned char> > horiSegments;
        unsigned char* ptrMark = foreImage.ptr<unsigned char>(currRect.y + i) + currRect.x;
        int length = horiEnd[i] - horiStart[i] + 1;
        findSegments(ptrMark + horiStart[i], length, horiSegments);
		if (horiSegments.size() == 1 && length > 20)
		{
			bottom = i;
			break;
		}
    }*/

    /*delete [] horiIsShadow;
    delete [] horiStart;
    delete [] horiEnd;*/

	if (top >= bottom)
		return Point(0, currRect.height - 1);
	else
		return Point(top, bottom);
}

Rect BlobExtractor::Impl::fitRectToForeground(const Mat& foreImage, const Rect& currRect, const Mat& transForeImage, const Rect& transCurrRect)
{
	int minTop = currRect.height;
	for (int i = 0; i < currRect.height; i++)
	{
		const unsigned char* ptrForeData = foreImage.ptr<unsigned char>(currRect.y + i) + currRect.x;
		for (int j = 0; j < currRect.width; j++)
		{
			if (ptrForeData[j] > 0)
			{
				minTop = i;
				break;
			}
		}
		if (minTop < currRect.height)
			break;
	}

	int maxBottom = -1;
	for (int i = currRect.height - 1; i >= 0; i--)
	{
		const unsigned char* ptrForeData = foreImage.ptr<unsigned char>(currRect.y + i) + currRect.x;
		for (int j = 0; j < currRect.width; j++)
		{
			if (ptrForeData[j] > 0)
			{
				maxBottom = i;
				break;
			}
		}
		if (maxBottom > -1)
			break;
	}

	int transMinTop = transCurrRect.height;
	for (int i = 0; i < transCurrRect.height; i++)
	{
		const unsigned char* ptrTransForeData = transForeImage.ptr<unsigned char>(transCurrRect.y + i) + transCurrRect.x;
		for (int j = 0; j < transCurrRect.width; j++)
		{
			if (ptrTransForeData[j] > 0)
			{
				transMinTop = i;
				break;
			}
		}
		if (transMinTop < transCurrRect.height)
			break;
	}

	int transMaxBottom = -1;
	for (int i = transCurrRect.height - 1; i >= 0; i--)
	{
		const unsigned char* ptrTransForeData = transForeImage.ptr<unsigned char>(transCurrRect.y + i) + transCurrRect.x;
		for (int j = 0; j < transCurrRect.width; j++)
		{
			if (ptrTransForeData[j] > 0)
			{
				transMaxBottom = i;
				break;
			}
		}
		if (transMaxBottom > -1)
			break;
	}

	Rect retRect;

	if (minTop == currRect.height || maxBottom == -1 ||
		transMinTop == transCurrRect.height || transMaxBottom == -1)
	{
		retRect = Rect(currRect.x, currRect.y, 0, 0);
	}
	else
	{
		retRect.x = currRect.x + transMinTop;
		retRect.y = currRect.y + minTop;
		retRect.width = transMaxBottom - transMinTop + 1;
		retRect.height = maxBottom - minTop + 1;
	}

	return retRect;
}

void BlobExtractor::Impl::refineObjectsDayModeNew(const Mat& normImage, const Mat& backImage, const Mat& gradDiffImage, 
	const vector<Object>& initObjects, vector<Object>& finalObjects)
{
    finalObjects.clear();
    if (initObjects.empty())
        return;

    Mat currObjectImage = Mat::zeros(gradDiffImage.rows, gradDiffImage.cols, CV_8UC1);
    Mat transCurrObjectImage = Mat::zeros(gradDiffImage.cols, gradDiffImage.rows, CV_8UC1);
    Mat transGradDiffImage = gradDiffImage.t();
	Mat transNormImage = normImage.t();
	Mat transBackImage = backImage.t();
    for (int i = 0; i < initObjects.size(); i++)
    {
        currObjectImage.setTo(0);
        drawContours(currObjectImage, initObjects[i].contours, -1, Scalar(255), -1);        
        Rect currRect = initObjects[i].rect;
		transCurrObjectImage = currObjectImage.t();
        Rect transCurrRect = Rect(currRect.y, currRect.x, currRect.height, currRect.width);
#if CMPL_WRITE_CONSOLE
        printf("Current Rect before refine: x = %3d, y = %3d, width = %3d, height = %3d\n",
                currRect.x, currRect.y, currRect.width, currRect.height);
#endif
		currRect = refineSingleObject(normImage, backImage, gradDiffImage, currObjectImage, currRect,
			transNormImage, transBackImage, transGradDiffImage, transCurrObjectImage, transCurrRect);
#if CMPL_WRITE_CONSOLE
		printf("Current Rect after refine:  x = %3d, y = %3d, width = %3d, height = %3d\n",
				currRect.x, currRect.y, currRect.width, currRect.height);
#endif
		if (currRect.width == 0 && currRect.height == 0)
		{
#if CMPL_WRITE_CONSOLE
			printf("Very similar to the back image, candidate object is filtered\n");
#endif
			continue;
		}
        Object currObject;
        currObject.rect = currRect & Rect(0, 0, normImage.cols, normImage.rows);
        currObject.contours = initObjects[i].contours;
        finalObjects.push_back(currObject);
    }
}

Rect BlobExtractor::Impl::refineSingleObject(const Mat& normImage, const Mat& backImage, const Mat& gradDiffImage, const Mat& foreImage, const Rect& currRect,
	const Mat& transNormImage, const Mat& transBackImage, const Mat& transGradDiffImage, const Mat& transForeImage, const Rect& transCurrRect)
{
	// 找水平方向的起止点和竖直方向的起止点
	vector<int> leftStart, rightEnd, topStart, bottomEnd;
	findPosStartAndEnd(foreImage, currRect, leftStart, rightEnd);
	findPosStartAndEnd(transForeImage, transCurrRect, topStart, bottomEnd);

	// 根据颜色和梯度信息找上下边界
	Point initTBByCG = 
		findRectOppositeBoundsByColorAndGrad(leftStart, rightEnd, 
		normImage, backImage, gradDiffImage, currRect);
	int top = initTBByCG.x;
	int bottom = initTBByCG.y;

	// 根据颜色信息找左右边界
	Point initLRByCG =
		findRectOppositeBoundsByColorAndGrad(topStart, bottomEnd, 
		transNormImage, transBackImage, transGradDiffImage, transCurrRect);
	int leftByColorAndGrad = initLRByCG.x;
	int rightByColorAndGrad = initLRByCG.y;
#if CMPL_WRITE_CONSOLE
	printf("refine by color and grad, left = %d, right = %d\n", leftByColorAndGrad, rightByColorAndGrad);
#endif
	// 根据形状信息找左右边界
	Point initLRByS =
		findRectOppositeBoundsByShape(topStart, bottomEnd, transNormImage, transCurrRect);
	int leftByShape = initLRByS.x;
	int rightByShape = initLRByS.y;
	// 下面对根据颜色和梯度找到的边界与之比较
	int centerByColorAndGrad = (leftByColorAndGrad + rightByColorAndGrad) / 2;	
	int centerByShape = (leftByShape + rightByShape) / 2;
	int centerDiff = abs(centerByColorAndGrad - centerByShape);   
	int lengthByColorAndGrad = rightByColorAndGrad - leftByColorAndGrad + 1;
	int lengthByShape = rightByShape - leftByShape + 1;	
	int intersectLength = min(rightByColorAndGrad, rightByShape) - max(leftByColorAndGrad, leftByShape);
	// 确定左右边界
	int left, right;
	// 如果通过颜色和梯度找到的边界和通过形状找到的边界差异较大，则认为通过形状信息找到的边界是准确的
	if (!(intersectLength > 0.5 * lengthByColorAndGrad && intersectLength > 0.5 * lengthByShape) ||
		centerDiff > 0.25 * currRect.height)
	{
#if CMPL_WRITE_CONSOLE
		printf("left right refine use shape result\n");
#endif
		left = leftByShape;
		right = rightByShape;
	}
	// 否则认为通过颜色和梯度信息找到的边界是准确的
	else
	{
		left = leftByColorAndGrad;
		right = rightByColorAndGrad;
	}

#if CMPL_WRITE_CONSOLE
	if (top != 0 || bottom != currRect.height - 1 || left != 0 || right != currRect.width - 1)
    {
        printf("Current Rect: x = %d, y = %d, width = %d, height = %d Changed in refineRectByGradient\n",
               currRect.x, currRect.y, currRect.width, currRect.height);
        printf("Orig Height: %3d. New vert boundaries: top: %d, bottom: %d\n", currRect.height, top, bottom);
        printf("Orig Width: %3d. New hori boundaries: left: %d, right: %d\n", currRect.width, left, right);
    }
#endif

    Rect retRect;
    // 如果计算出新矩形
    if (left > right || top > bottom)
    {
        retRect = Rect(0, 0, 0, 0);
    }
    // 否则对新的矩形边界进行判决，输出新的矩形
    else
    {
        top = 0;
        if (left < 0.075 * currRect.width)
            left = 0;
        if (right > 0.925 * currRect.width)
            right = currRect.width;
        else
            right++;
        if (top < 0.075 * currRect.height)
            top = 0;
        if (bottom > 0.925 * currRect.height)
            bottom = currRect.height;
        else
            bottom += 5;
        retRect.x = currRect.x + left;
        retRect.y = currRect.y + top;
        retRect.width = right - left;
        retRect.height = bottom - top;
    }

    return retRect;
}

void BlobExtractor::Impl::findPosStartAndEnd(const Mat& foreImage, const Rect& currRect, vector<int>& posStart, vector<int>& posEnd)
{
	int initHoriStart = currRect.width;
    int initHoriEnd = -1;
	posStart.assign(currRect.height, initHoriStart);
	posEnd.assign(currRect.height, initHoriEnd);

    // 查找每一行的起始编号和终止编号
    for (int i = 0; i < currRect.height; i++)
    {
        const unsigned char* ptrMark = foreImage.ptr<unsigned char>(currRect.y + i) + currRect.x;
        //float* ptrMark = gradDiffImage.ptr<float>(currRect.y + i) + currRect.x;
        for (int j = 0; j < currRect.width; j++)
        {
            if (ptrMark[j] > 0)
            {
                posStart[i] = j;
                break;
            }
        }
    }
    for (int i = 0; i < currRect.height; i++)
    {
        const unsigned char* ptrMark = foreImage.ptr<unsigned char>(currRect.y + i) + currRect.x;
        //float* ptrMark = gradDiffImage.ptr<float>(currRect.y + i) + currRect.x;
        for (int j = currRect.width - 1; j >= 0; j--)
        {
            if (ptrMark[j] > 0)
            {
                posEnd[i] = j;
                break;
            }
        }
    }
}

Point BlobExtractor::Impl::findRectOppositeBoundsByColorAndGrad(const vector<int>& posStart, const vector<int>& posEnd, 
	const Mat& normImage, const Mat& backImage, const Mat& gradDiffImage, const Rect& currRect)
{
    int initHoriStart = currRect.width;
    int initHoriEnd = -1;
	
	// 标记每一行是否为候选阴影行
    //unsigned char* horiIsShadow = new unsigned char [currRect.height];
	vector<unsigned char> horiIsShadow(currRect.height);
    for (int i = 0; i < currRect.height; i++)
    {
        if (posStart[i] == initHoriStart && posEnd[i] == initHoriEnd/* ||
            horiEnd[i] - horiStart[i] < 0.1 * currRect.width*/)
        {
            horiIsShadow[i] = 1;
            continue;
        }

		int length = posEnd[i] - posStart[i] + 1;
        horiIsShadow[i] = 0;

		vector<Segment<unsigned char> > horiSegments;
        const unsigned char* ptrMark = gradDiffImage.ptr<unsigned char>(currRect.y + i) + currRect.x;
        findSegments(ptrMark + posStart[i], length, horiSegments);
		/*if (horiSegments.size() == 1 && horiSegments[0].data == 0 && 
			(horiSegments[0].length < 20 || horiSegments[0].length < 0.1 * currRect.height))
		{
			horiIsShadow[i] = 1;
			continue;
		}*/

		bool isShadByGradDiff = false;		
        for (int j = 0; j < horiSegments.size(); j++)
        {
            if (horiSegments[j].data == 0 &&
				(horiSegments[j].length > 0.6 * length || 
				 horiSegments.size() > j + 2 && horiSegments[j + 1].length < 10 && 
				 horiSegments[j].length + horiSegments[j + 1].length + horiSegments[j + 2].length > 0.7 * length))
			{				
				isShadByGradDiff = true;
				break;
            }
        }

		int lineShadCount = 0;
		const unsigned char* ptrNormData = normImage.ptr<unsigned char>(currRect.y + i) + currRect.x * 3;
		const unsigned char* ptrBackData = backImage.ptr<unsigned char>(currRect.y + i) + currRect.x * 3;
		for (int j = posStart[i]; j <= posEnd[i]; j++)
		{
			if (ptrNormData[j * 3] < ptrBackData[j * 3] &&
				ptrNormData[j * 3 + 1] < ptrBackData[j * 3 + 1] &&
				ptrNormData[j * 3 + 2] < ptrBackData[j * 3 + 2])
			{
				unsigned char maxVal;
				maxVal = ptrNormData[j * 3] > ptrNormData[j * 3 + 1] ? ptrNormData[j * 3] : ptrNormData[j * 3 + 1];
				maxVal = ptrNormData[j * 3 + 2] > maxVal ? ptrNormData[j * 3 + 2] : maxVal;
				if (maxVal == 0)
				{
					lineShadCount++;
					continue;
				}
				unsigned char minVal;
				minVal = ptrNormData[j * 3] < ptrNormData[j * 3 + 1] ? ptrNormData[j * 3] : ptrNormData[j * 3 + 1];
				minVal = ptrNormData[j * 3 + 2] < minVal ? ptrNormData[j * 3 + 2] : minVal;
				if ((maxVal - minVal) < 0.2 * maxVal)
					lineShadCount++;
			}
		}
		bool isShadByColor = lineShadCount > length * 0.7;        
		horiIsShadow[i] = isShadByColor && isShadByGradDiff;
    }
	localMedian(horiIsShadow, 5);
	//destroyWindow("is shadow");
	//showArrayByVertBar("is shadow", horiIsShadow, false, true);
	//waitKey(0);

	int topByColorAndGrad = 0, bottomByColorAndGrad = currRect.height - 1;

    // 给判决结果数组 horiIsShadow 分段
    vector<Segment<unsigned char> > horiSegmentsByColorAndGrad;
    findSegments(horiIsShadow, horiSegmentsByColorAndGrad);	
    // 找新的竖直方向上的边界点
    for (int i = 0; i < horiSegmentsByColorAndGrad.size(); i++)
    {
		if (horiSegmentsByColorAndGrad[i].begin < int(currRect.height * 0.1) &&
			horiSegmentsByColorAndGrad[i].length > int(currRect.height * 0.15) &&
            horiSegmentsByColorAndGrad[i].data ||
			horiSegmentsByColorAndGrad[i].begin < int(currRect.height * 0.3) &&
			horiSegmentsByColorAndGrad[i].length > int(currRect.height * 0.2) &&
            horiSegmentsByColorAndGrad[i].data )
        {
            topByColorAndGrad = horiSegmentsByColorAndGrad[i].end;
            break;
        }
    }
    for (int i = horiSegmentsByColorAndGrad.size() - 1; i >= 0; i--)
    {
		if (horiSegmentsByColorAndGrad[i].end > int(currRect.height * 0.9) &&
			horiSegmentsByColorAndGrad[i].length > int(currRect.height * 0.15) &&
            horiSegmentsByColorAndGrad[i].data ||
			horiSegmentsByColorAndGrad[i].end > int(currRect.height * 0.7) &&
			horiSegmentsByColorAndGrad[i].length > int(currRect.height * 0.2) &&
            horiSegmentsByColorAndGrad[i].data)
        {
            bottomByColorAndGrad = horiSegmentsByColorAndGrad[i].begin;
            break;
        }
	}

	/*if (topByColorAndGrad > bottomByColorAndGrad ||
		bottomByColorAndGrad - topByColorAndGrad < 0.3 * currRect.height)
		return Point(0, currRect.height - 1);
	else*/
		return Point(topByColorAndGrad, bottomByColorAndGrad);
}

Point BlobExtractor::Impl::findRectOppositeBoundsByShape(const vector<int>& posStart, const vector<int>& posEnd, const Mat& normImage, const Rect& currRect)
{
	// 计算每行长度的平均值
	vector<int> rowLength(currRect.height, 0);
	double avgLength = 0;
	for (int i = 0; i < currRect.height; i++)
	{
		rowLength[i] = currRect.width - posStart[i];
		// rowLength[i] = posEnd[i] - posStart[i];
		avgLength += rowLength[i];
	}
	localMedian(rowLength, 5);
	localMean(rowLength, 5);
	avgLength /= currRect.height;
#if CMPL_SHOW_IMAGE
	showArrayByVertBar("col mean", rowLength, true, true);
#endif

	// 判断每行的长度是否大于长度的均值
	vector<unsigned char> isRowWide(currRect.height);
	for (int i = 0; i < currRect.height; i++)
	{
		isRowWide[i] = rowLength[i] > avgLength ? 1 : 0;
	}
	localMedian(isRowWide, 5);

	// 找连续段
	vector<Segment<unsigned char> > horiSegmentsByShape;
	findSegments(isRowWide, horiSegmentsByShape);
	
	// 找长度大于均值，长度最长的段
	int maxIndex = 0, maxLength = 0;
	for (int i = 0; i < horiSegmentsByShape.size(); i++)
	{
		if (horiSegmentsByShape[i].data > 0 && horiSegmentsByShape[i].length > maxLength)
		{
			maxIndex = i;
			maxLength = horiSegmentsByShape[i].length;
		}
	}

	// 取上面找到的段的两个端点，计算这两个端点邻域内的差值	
	int winLength = 21;
	int temp;
	int currMin, currMax;

	temp = horiSegmentsByShape[maxIndex].begin - winLength / 2;
	int beginBegin = temp > 0 ? temp : 0;
	temp = horiSegmentsByShape[maxIndex].begin + winLength / 2;
	int beginEnd = temp < currRect.height - 1 ? temp : currRect.height - 1;	
	currMin = currMax = rowLength[beginBegin];
	for (int i = beginBegin + 1; i <= beginEnd; i++)
	{
		currMin = currMin < rowLength[i] ? currMin : rowLength[i];
		currMax = currMax > rowLength[i] ? currMax : rowLength[i];
	}
	int beginMaxDiff = currMax - currMin;

	temp = horiSegmentsByShape[maxIndex].end - winLength / 2;
	int endBegin = temp > 0 ? temp : 0;
	temp = horiSegmentsByShape[maxIndex].end + winLength / 2;
	int endEnd = temp < currRect.height - 1 ? temp : currRect.height - 1;
	currMin = currMax = rowLength[endBegin];
	for (int i = endBegin + 1; i <= endEnd; i++)
	{
		currMin = currMin < rowLength[i] ? currMin : rowLength[i];
		currMax = currMax > rowLength[i] ? currMax : rowLength[i];
	}
	int endMaxDiff = currMax - currMin;

#if CMPL_WRITE_CONSOLE
	printf("max length segment, begin: %d, end: %d\n", 
		horiSegmentsByShape[maxIndex].begin, horiSegmentsByShape[maxIndex].end);
	printf("beginMaxDiff: %d, endMaxDiff: %d\n", beginMaxDiff, endMaxDiff);
#endif

	int topByShape = 0, bottomByShape = currRect.height;
	// 如果 horiSegments[maxIndex] 两个端点邻域内最大差异值足够大	
	if ((currRect.x < 10 || normImage.cols - currRect.x < 10) && 
		(beginMaxDiff > 0.1 * currRect.width || beginMaxDiff > 20) && 
		(endMaxDiff > 0.1 * currRect.width || endMaxDiff > 20) ||
		currRect.x >= 10 && beginMaxDiff > 25 && endMaxDiff > 25 )
	{
		// 则根据这两个端点确定由形状信息得到的边界
		int center = currRect.height / 2;
		int segCenter = (horiSegmentsByShape[maxIndex].begin + horiSegmentsByShape[maxIndex].end) / 2;
		if (abs(segCenter - center) > currRect.height * 0.2 ||
			horiSegmentsByShape[maxIndex].begin > 15 || horiSegmentsByShape[maxIndex].begin > 0.2 * currRect.height ||
			currRect.height - horiSegmentsByShape[maxIndex].end > 15 || horiSegmentsByShape[maxIndex].end < 0.8 * currRect.height)
		{
			//topByShape = horiSegmentsByShape[maxIndex].begin - horiSegmentsByShape[maxIndex].length * 0.05;
			//topByShape = topByShape < 0 ? 0 : topByShape;
			//bottomByShape = horiSegmentsByShape[maxIndex].end + horiSegmentsByShape[maxIndex].length * 0.05;
			//bottomByShape = bottomByShape >= currRect.height - 1 ? currRect.height - 1 : bottomByShape;
			topByShape = horiSegmentsByShape[maxIndex].begin;
			bottomByShape = horiSegmentsByShape[maxIndex].end;
		}
	}

	return Point(topByShape, bottomByShape);
}

} // end namespace zsfo