#pragma once

#include <string>
#include <vector>
#include <list>

#include <opencv2/core/core.hpp>

#include "MovingObjectDetector.h"
#include "RegionOfInterest.h"
#include "BlobTracker.h"

namespace zsfo
{

struct StaticBlob
{
	StaticBlob(void) {ID = 0;};
	StaticBlob(const StaticBlob& instance) {ID = 0; configCheckStatic = instance.configCheckStatic;}; 
	~StaticBlob(void) {};

    void init(const std::string& path);
	void init(int currID, const cv::Rect& currRect, long long int time);
	void setConfigParam(const double* minStaticTimeInMinute = 0);
	void checkStatic(const long long int time, const int count);
	void outputInfo(StaticObjectInfo& objectInfo) const;
	void drawBlob(cv::Mat& image, const cv::Scalar& staticColor, const cv::Scalar& nonStaticColor) const;
	void displayHistory(void) const;

	int ID;
	cv::Rect rect;
	bool isStatic;
	bool hasOutputStatic;
	struct Interval
	{
		Interval() {};
		Interval(long long int time, bool match) {beg = end = time, isMatch = match;};
		long long int beg, end;
		bool isMatch;
	};
	std::vector<Interval> intervals;
	struct ConfigCheckStatic
	{
		double minStaticTimeInMinute;
	};
	cv::Ptr<ConfigCheckStatic> configCheckStatic;
};

class StaticBlobTracker::Impl
{
public:
	Impl(void) {};
	~Impl(void) {};

	void init(const RegionOfInterest& observedRegion, const SizeInfo& sizesOrigAndNorm, const std::string& path);
	void setConfigParam(const double* allowedMissTimeInMinute = 0, const double* minStaticTimeInMinute = 0);
	void proc(const long long int time, const int count, 
		const std::vector<cv::Rect>& rects, std::vector<StaticObjectInfo>& staticObjects);	
	void drawBlobs(cv::Mat& image, const cv::Scalar& staticColor, const cv::Scalar& nonStaticColor) const;

private:
	void updateBlobList(const long long int time, const int count, const std::vector<cv::Rect>& rects);
	void checkStatic(const long long int time, const int count);
	void outputInfo(std::vector<StaticObjectInfo>& staticObjects) const;

	int blobCount;
	std::list<StaticBlob*> blobList;
	RegionOfInterest roi;
	SizeInfo sizeInfo;
	StaticBlob blobInstance;
	struct ConfigUpdateBlobList
	{
		double minMissTimeInMinuteToDelete;
	};
	ConfigUpdateBlobList configUpdateBlobList;

	struct RectInfo
	{
		RectInfo(void) {};
		RectInfo(const cv::Rect& rct) {rect = rct; isMatch = false;};
		cv::Rect rect;
		bool isMatch;
	};
};

} // namespace zsfo