#include <cmath>
#include <sstream>
#include <opencv2/imgproc/imgproc.hpp>
#include "RegionOfInterest.h"
#include "FileStreamScopeGuard.h"
#include "Exception.h" 
#include "CompileControl.h"

namespace zsfo
{

RegionOfInterest::RegionOfInterest(const RegionOfInterest& roi)
    : isFullSize(roi.isFullSize), width(roi.width), height(roi.height),
      imageRect(roi.imageRect), mask(roi.mask.clone()), 
      defIncRegion(roi.defIncRegion), vertices(roi.vertices)
{}

RegionOfInterest& RegionOfInterest::operator=(const RegionOfInterest& roi)
{
    if (this == &roi) return *this;
    isFullSize = roi.isFullSize;
    width = roi.width;
    height = roi.height;
    imageRect = roi.imageRect;
    mask = roi.mask.clone();
    defIncRegion = roi.defIncRegion;
    vertices = roi.vertices;
    return *this;
}

void RegionOfInterest::init(const cv::Size& imageSize, const std::string& path, const std::string& label)
{
	std::fstream initFileStream;
    ztool::FileStreamScopeGuard<std::fstream> guard(initFileStream);
	initFileStream.open(path.c_str());
	if (!initFileStream.is_open())
	{
        THROW_EXCEPT("cannot open file " + path);
	}
	std::string buf;
	char strNotUsed[500];
	while (true)
	{
		if (initFileStream.eof())
		{
            THROW_EXCEPT("cannot find config params label " + label);
		}
		std::getline(initFileStream, buf);
		if (buf.find(label) != std::string::npos)
			break;
	}
    while (true)
	{
		if (initFileStream.eof())
		{
            THROW_EXCEPT("cannot find config params label #define_included_region");
		}
		std::getline(initFileStream, buf);
		if (buf.find("#define_included_region") != std::string::npos)
			break;
	}    
	bool includedRegion;
    {
        std::stringstream strm(buf);
	    strm >> strNotUsed >> includedRegion;
    }
	while (true)
	{
		if (initFileStream.eof())
		{
            THROW_EXCEPT("cannot find config params label #num_of_roi");
		}
		std::getline(initFileStream, buf);
		if (buf.find("#num_of_roi") != std::string::npos)
			break;
	}
	int numOfROI;
    {
        std::stringstream strm(buf);
        strm >> strNotUsed >> numOfROI;
    }
	if (numOfROI == 0)
	{
		init(label, imageSize, includedRegion, std::vector<std::vector<cv::Point> >());
		return;
	}
	std::vector<std::vector<cv::Point> > points(numOfROI);
	for (int i = 0; i < numOfROI; i++)
	{
		while (true)
		{
			if (initFileStream.eof())
			{
				std::stringstream strm;
				strm << "cannot find enough number of ROI as many as " << numOfROI;
				THROW_EXCEPT(strm.str());
			}
			std::getline(initFileStream, buf);
			if (buf.find("#roi") != std::string::npos)
				break;
		}
		std::stringstream strm(buf);
		std::vector<int> vals;
		strm >> strNotUsed;
		while (!strm.eof())
		{
			int val;
			strm >> val;
			vals.push_back(val);
		}
		int size = vals.size();
		if (size == 0 || size % 2 == 1)
		{
            THROW_EXCEPT("invalid number of values following label #roi");
		}
		size /= 2;
		points[i].resize(size);
		for (int j = 0; j < size; j++)
			points[i][j] = cv::Point(vals[j * 2], vals[j * 2 + 1]);
	}
	init(label, imageSize, includedRegion, points);
};

void RegionOfInterest::init(const std::string& label, const cv::Size& imageSize, bool defIncludedRegion,
    const std::vector<std::vector<cv::Point> >& imagePoints)
{
	if (imageSize.width <= 0 || imageSize.height <= 0)
	{
		std::stringstream sstrm;
		sstrm << "imageSize.width = " << imageSize.width << ", "
            << "imageSize.height = " << imageSize.height << ", unsupported";
		THROW_EXCEPT(sstrm.str());
	}
    defIncRegion = defIncludedRegion;
	if (imagePoints.empty())
	{
		isFullSize = defIncRegion;
		width = imageSize.width;
		height = imageSize.height;
		imageRect = cv::Rect(0, 0, width, height);
		vertices.resize(1);
		vertices[0].resize(4);
		vertices[0][0] = cv::Point(0, 0);
		vertices[0][1] = cv::Point(0, height);
		vertices[0][2] = cv::Point(width, height);
		vertices[0][3] = cv::Point(width, 0);
		mask.create(height, width, CV_8UC1);        
		mask.setTo(defIncRegion ? 255 : 0);
	}
    else if (imagePoints.size() == 1 && imagePoints[0].size() < 2)
    {
        THROW_EXCEPT("imagePoints.size == 1 && imagePoints[0].size() < 2");
    }
    else if (imagePoints.size() == 1 && imagePoints[0].size() == 2)
    {
        if (!defIncludedRegion)
        {
            THROW_EXCEPT("imagePoints.size == 1 && imagePoints[0].size() == 2 && !defIncludedRegion");
        }
        width = imageSize.width;
		height = imageSize.height;
		imageRect = cv::Rect(0, 0, width, height);
		vertices = imagePoints;
		mask.create(height, width, CV_8UC1);
        mask.setTo(0);
        cv::line(mask, imagePoints[0][0], imagePoints[0][1], 255, 1);
        cv::Mat kern = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(40, 40));
        // 线段向两侧扩展 40 像素宽度作为实际的感兴趣区域
        cv::dilate(mask, mask, kern);
        isFullSize = false;
    }
    else
	{
		int vectorSize = imagePoints.size();
		cv::Rect rect(0, 0, imageSize.width, imageSize.height);
		for (int i = 0; i < vectorSize; i++)
		{
			int size = imagePoints[i].size();
			if (size < 3)
			{
				std::stringstream strm;
				strm << "ERROR in function " << __FUNCTION__ << "(), "
					 << "imagePoints[" << i << "].size() < 3";
                THROW_EXCEPT(strm.str());
			}
		}
		width = imageSize.width;
		height = imageSize.height;
		imageRect = cv::Rect(0, 0, width, height);
		vertices = imagePoints;
		mask.create(height, width, CV_8UC1);
        if (defIncludedRegion)
        {
		    mask.setTo(0);
		    for (int i = 0; i < vectorSize; i++)
		    {
			    std::vector<std::vector<cv::Point> > tempPoints(1);
			    tempPoints[0] = imagePoints[i];
			    cv::drawContours(mask, tempPoints, -1, cv::Scalar(255, 255, 255), -1);
		    }
        }
        else
        {
            mask.setTo(255);
		    for (int i = 0; i < vectorSize; i++)
		    {
			    std::vector<std::vector<cv::Point> > tempPoints(1);
			    tempPoints[0] = imagePoints[i];
			    cv::drawContours(mask, tempPoints, -1, cv::Scalar(0, 0, 0), -1);
		    }
        }
		isFullSize = (double(cv::countNonZero(mask)) > 0.99 * width * height);
	}
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
    printf("display region of interest %s:\n", label.c_str());
    printf("  define included roi: %s\n", defIncRegion ? "true" : "false");
    printf("  number of roi: %d\n", vertices.size());
    int numOfPoly = vertices.size();
	for (int i = 0; i < numOfPoly; i++)
	{
		printf("  roi %d: ", i + 1);
		int size = vertices[i].size();
		for (int j = 0; j < size; j++)
		{
			printf("(%d, %d) ", vertices[i][j].x, vertices[i][j].y);
		}
		printf("\n");
	}
	printf("\n");
#endif
};

void RegionOfInterest::filterNotIntersecting(std::vector<cv::Rect>& rects) const
{
	if (rects.empty()) return;
	int size = rects.size();
	std::vector<cv::Rect> temp;
	temp.reserve(size);	
	for (int i = 0; i < size; i++)
	{
		if (intersects(rects[i]))
			temp.push_back(rects[i]);
	}
    rects.swap(temp);
};

void RegionOfInterest::filterNotIntersecting(std::vector<cv::Rect>& rects, const double ratio) const
{
	if (rects.empty()) return;
	int size = rects.size();
	std::vector<cv::Rect> temp;
	temp.reserve(size);	
	for (int i = 0; i < size; i++)
	{
		if (intersects(rects[i], ratio))
			temp.push_back(rects[i]);
	}
    rects.swap(temp);
};

const static int minSideLen = 10;
const static int scaleSize = 11;
const static float scales[scaleSize] = {0, 0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F, 0.8F, 0.9F, 1.0F};

bool RegionOfInterest::intersects(const cv::Rect& testRect) const
{
	cv::Rect rect = testRect & imageRect;
	if (isFullSize && rect.width > 0 && rect.height > 0) return true;
	if (rect.width == 0 || rect.height == 0) return false;
	cv::Point tl = rect.tl();
	if (contains(tl)) return true;
	cv::Point tr = tl + cv::Point(rect.width - 1, 0);
	if (contains(tr)) return true;
	cv::Point bl = tl + cv::Point(0, rect.height - 1);
	if (contains(bl)) return true;
	cv::Point br = tl + cv::Point(rect.width - 1, rect.height -1);
	if (contains(br)) return true;

	if (rect.width <= minSideLen * 2 && rect.height <= minSideLen * 2)
		return false;

	int begX = tl.x;
	int begY = tl.y;
	int endX = br.x;
	int endY = br.y;

	if (rect.width <= minSideLen * 2 && rect.height > minSideLen * 2)
	{
		for (int i = 0; i < scaleSize; i++)
		{
			int currY = begY * scales[scaleSize - 1 - i] + endY * scales[i];
			const unsigned char* ptrMask = mask.ptr<unsigned char>(currY);
			if (ptrMask[begX]) return true;
			if (ptrMask[endX]) return true;
		}
	}
	else if (rect.width > minSideLen * 2 && rect.height <= minSideLen * 2)
	{
		const unsigned char* ptrMask1 = mask.ptr<unsigned char>(begY);
		const unsigned char* ptrMask2 = mask.ptr<unsigned char>(endY);
		for (int j = 0; j < scaleSize; j++)
		{
			if (ptrMask1[int(begX * scales[scaleSize - 1 - j] + endX * scales[j])])
				return true;
			if (ptrMask2[int(begX * scales[scaleSize - 1 - j] + endX * scales[j])])
				return true;
		}
	}
	else
	{
		for (int i = 0; i < scaleSize; i++)
		{
			int currY = begY * scales[scaleSize - 1 - i] + endY * scales[i];
			const unsigned char* ptrMask = mask.ptr<unsigned char>(currY);
			for (int j = 0; j < scaleSize; j++)
			{
				if (ptrMask[int(begX * scales[scaleSize - 1 - j] + endX * scales[j])])
					return true;
			}
		}
	}
	return false;
};

bool RegionOfInterest::intersects(const cv::Rect& testRect, const double ratio) const
{
	cv::Rect rect = testRect & imageRect;
	if (isFullSize && rect.width > 0 && rect.height > 0) return true;
	if (rect.width == 0 || rect.height == 0) return false;
	double count = 0;
	cv::Point tl = rect.tl();
	if (contains(tl)) count++;
	cv::Point tr = tl + cv::Point(rect.width - 1, 0);
	if (contains(tr)) count++;
	cv::Point bl = tl + cv::Point(0, rect.height - 1);
	if (contains(bl)) count++;
	cv::Point br = tl + cv::Point(rect.width - 1, rect.height -1);
	if (contains(br)) count++;

	if (rect.width <= minSideLen && rect.height <= minSideLen)
		return double(count) > 4 * ratio;

	int begX = tl.x;
	int begY = tl.y;
	int endX = br.x;
	int endY = br.y;

	count = 0;
	if (rect.width <= minSideLen && rect.height > minSideLen)
	{
		int thresCount = ratio * scaleSize * 2;
		for (int i = 0; i < scaleSize; i++)
		{
			int currY = begY * scales[scaleSize - 1 - i] + endY * scales[i];
			const unsigned char* ptrMask = mask.ptr<unsigned char>(currY);
			if (ptrMask[begX]) count++;
			if (ptrMask[endX]) count++;
			if (double(count) > thresCount)
				return true;
		}
		return double(count) > thresCount;
	}
	else if (rect.width > minSideLen && rect.height <= minSideLen)
	{
		int thresCount = ratio * scaleSize * 2;
		const unsigned char* ptrMask1 = mask.ptr<unsigned char>(begY);
		const unsigned char* ptrMask2 = mask.ptr<unsigned char>(endY);
		for (int j = 0; j < scaleSize; j++)
		{
			if (ptrMask1[int(begX * scales[scaleSize - 1 - j] + endX * scales[j])])
				count++;
			if (ptrMask2[int(begX * scales[scaleSize - 1 - j] + endX * scales[j])])
				count++;
			if (double(count) > thresCount)
				return true;
		}
		return double(count) > thresCount;
	}
	else
	{
		int thresCount = ratio * scaleSize * scaleSize;
		for (int i = 0; i < scaleSize; i++)
		{
			int currY = begY * scales[scaleSize - 1 - i] + endY * scales[i];
			const unsigned char* ptrMask = mask.ptr<unsigned char>(currY);
			for (int j = 0; j < scaleSize; j++)
			{
				if (ptrMask[int(begX * scales[scaleSize - 1 - j] + endX * scales[j])])
					count++;
				if (double(count) > thresCount)
				return true;
			}
		}
		return double(count) > thresCount;
	}
	return false;
};

void RegionOfInterest::draw(cv::Mat& image, const cv::Scalar& color) const
{
	int numOfVector = vertices.size();
	for (int i = 0; i < numOfVector; i++)
	{
		int size = vertices[i].size();
		for (int j = 1; j < size; j++)
			cv::line(image, vertices[i][j - 1], vertices[i][j], color, 2);
		if (size == 2)
			break;
		cv::line(image, vertices[i][0], vertices[i][size - 1], color, 2);
	}
    if (!defIncRegion)
    {
        cv::line(image, cv::Point(0, 0), cv::Point(0, height), color, 3);
        cv::line(image, cv::Point(0, height), cv::Point(width, height), color, 3);
        cv::line(image, cv::Point(width, height), cv::Point(width, 0), color, 3);
        cv::line(image, cv::Point(0, 0), cv::Point(width, 0), color, 3);
    }
};

void VirtualLoop::init(const std::string& path, const std::string& label)
{
	std::fstream initFileStream;
    ztool::FileStreamScopeGuard<std::fstream> guard(initFileStream);
    initFileStream.open(path.c_str());
	if (!initFileStream.is_open())
	{
        THROW_EXCEPT("cannot open file " + path);
	}
	char stringNotUsed[500];
	do
	{
		initFileStream >> stringNotUsed;
		if (initFileStream.eof())
		{
            THROW_EXCEPT("cannot find config params label " + label);
		}
	}
	while(std::string(stringNotUsed) != label);

    std::vector<cv::Point> points(4);
    std::vector<unsigned int> legalDir(4);
    int numOfLane;
	initFileStream >> stringNotUsed;
	initFileStream >> points[0].x >> points[0].y
				   >> points[1].x >> points[1].y
				   >> points[2].x >> points[2].y
				   >> points[3].x >> points[3].y;   // 图像坐标
	initFileStream.close();

    init(label, points);
}

void VirtualLoop::init(const std::string& label, const std::vector<cv::Point>& points)
{
    if (points.size() != 4)
        THROW_EXCEPT("imagePoints.size() != 4");

    for (int i = 0; i < 4; i++)
        vertices[i] = points[i];

#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
	printf("display VirtualLoop %s\n", label.c_str());
	printf("  points: ");
	for (int i = 0; i < 4; i++)
		printf("(%d, %d) ", vertices[i].x, vertices[i].y);
	printf("\n");
#endif

	leftK = (double)(vertices[1].x - vertices[0].x) /
        (double)(vertices[1].y - vertices[0].y);
    leftB = (double)(vertices[0].x * vertices[1].y - vertices[1].x * vertices[0].y) /
        (double)(vertices[1].y - vertices[0].y);
    rightK = (double)(vertices[3].x - vertices[2].x) /
        (double)(vertices[3].y - vertices[2].y);
    rightB = (double)(vertices[2].x * vertices[3].y - vertices[3].x * vertices[2].y) /
        (double)(vertices[3].y - vertices[2].y);
    topK = (double)(vertices[3].y - vertices[0].y) /
        (double)(vertices[3].x - vertices[0].x);
    topB = (double)(vertices[3].x * vertices[0].y - vertices[0].x * vertices[3].y) /
        (double)(vertices[3].x - vertices[0].x);
    bottomK = (double)(vertices[2].y - vertices[1].y) /
        (double)(vertices[2].x - vertices[1].x);
    bottomB = (double)(vertices[2].x * vertices[1].y - vertices[1].x * vertices[2].y) /
        (double)(vertices[2].x - vertices[1].x);
}

bool VirtualLoop::intersects(const cv::Rect& testRect) const
{
    // 先检测顶点，四边的中点和矩形的中心是否在线圈内
	if (contains(cv::Point(testRect.x, testRect.y))) return true;
    if (contains(cv::Point(testRect.x + testRect.width, testRect.y))) return true;
    if (contains(cv::Point(testRect.x, testRect.y + testRect.height))) return true;
    if (contains(cv::Point(testRect.x + testRect.width, testRect.y + testRect.height))) return true;
    if (contains(cv::Point(testRect.x + testRect.width / 2, testRect.y))) return true;
    if (contains(cv::Point(testRect.x + testRect.width / 2, testRect.y + testRect.height))) return true;
    if (contains(cv::Point(testRect.x, testRect.y + testRect.height / 2))) return true;
    if (contains(cv::Point(testRect.x + testRect.width, testRect.y + testRect.height / 2))) return true;
    if (contains(cv::Point(testRect.x + testRect.width / 2, testRect.y + testRect.height / 2))) return true;

	if (testRect.width < 10 || testRect.height < 10) return false;

	int horiStep = ceil(double(testRect.width) / 10);
    int vertStep = ceil(double(testRect.height) / 10);
    for (int i = vertStep; i < testRect.height; i = i + vertStep)
    {
        for (int j = horiStep; j < testRect.width; j = j + horiStep)
        {
            if (contains(cv::Point(testRect.x + j, testRect.y + i)))
            {
                return true;
            }
        }
    }
    return false;
}

void LineSegment::init(const std::string& path, const std::string& label)
{
    std::fstream initFileStream;
    ztool::FileStreamScopeGuard<std::fstream> guard(initFileStream);
	initFileStream.open(path.c_str());
	if (!initFileStream.is_open())
	{
        THROW_EXCEPT("cannot open file " + path);
	}
	std::string buf;
	char strNotUsed[500];
	while (true)
	{
		if (initFileStream.eof())
		{
            THROW_EXCEPT("cannot find config params label " + label);
		}
		std::getline(initFileStream, buf);
		if (buf.find(label) != std::string::npos)
			break;
	}
    cv::Point begPoint, endPoint, begSidePoint;
    initFileStream >> buf >> begPoint.x >> begPoint.y;
    initFileStream >> buf >> endPoint.x >> endPoint.y;
    initFileStream >> buf >> begSidePoint.x >> begSidePoint.y;
    initFileStream.close();
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
    printf("display line segment:\n");
    printf("  begPoint = (%d, %d)\n", begPoint.x, begPoint.y);
    printf("  endPoint = (%d, %d)\n", endPoint.x, endPoint.y);
    printf("  begSidePoint = (%d, %d)\n", begSidePoint.x, begSidePoint.y);
    printf("\n");
#endif
    init(begPoint, endPoint, begSidePoint);
}

void LineSegment::init(const cv::Point& begPoint, const cv::Point& endPoint, 
    const cv::Point& begSidePoint)
{
    if (begPoint == endPoint)
        THROW_EXCEPT("begPoint == endPoint");
    beg = begPoint;
    end = endPoint;
    if (beg.x == end.x)
    {
        //region = cv::Rect(beg.x, std::min(beg.y, end.y), 1, std::abs(beg.y - end.y));
        //region = Circle(beg, end);
        region = ParallelLineInterior(beg, end);
        a = 1;
        b = 0;
        c = -beg.x;
        signOnBegSide = begSidePoint.x * a + begSidePoint.y * b + c > 0 ? 1 : -1;
        return;
    }
    else if (beg.y == end.y)
    {
        //region = cv::Rect(std::min(beg.x, end.x), beg.y, std::abs(beg.x - end.x), 1);
        //region = Circle(beg, end);
        region = ParallelLineInterior(beg, end);
        a = 0;
        b = 1;
        c = -beg.y;
        signOnBegSide = begSidePoint.x * a + begSidePoint.y * b + c > 0 ? 1 : -1;
        return;
    }
    
    //region = cv::Rect(beg, end);
    //region = Circle(beg, end);
    region = ParallelLineInterior(beg, end);
    double kk = double(beg.y - end.y) / double(beg.x - end.x);
    double bb = beg.y - kk * beg.x;
    double ss = sqrt(kk * kk + 1.0);
    a = kk / ss;
    b = -1.0 / ss;
    c = bb / ss;
    double val = begSidePoint.x * a + begSidePoint.y * b + c;
    if (std::abs(val) < 0.01 && begSidePoint.x >= 0 && begSidePoint.y >= 0)
        THROW_EXCEPT("begSidePoint too close to line");
    signOnBegSide = val > 0 ? 1 : -1;
}

}