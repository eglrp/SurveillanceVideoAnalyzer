#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <opencv2/imgproc/imgproc.hpp>
#include "ProcVideo.h"
#include "VideoSplitter.h"
#include "OperateData.h"
#include "ShowData.h"
#include "Segment.h"
#include "CreateDirectory.h"
#include "Exception.h"

using namespace std;
using namespace cv;
using namespace zsfo;
using namespace ztool;

namespace zvs
{

VideoAnalyzer::VideoAnalyzer()
{

}

VideoAnalyzer::~VideoAnalyzer()
{

}

void VideoAnalyzer::init(Mat& image)
{
	imageWidth = image.cols;
	imageHeight = image.rows;
	pixSum = imageWidth * imageHeight;
	
	medianBlur(image, blurImage, 3);
	GaussianBlur(blurImage, blurImage, Size(3, 3), 0.0);
	cvtColor(blurImage, grayImage, CV_BGR2GRAY);
	//calcThresholdedGradient(grayImage, gradImage, 200);
	calcGradient(grayImage, gradImage);
	medianBlur(gradImage, gradImage, 3);
	GaussianBlur(gradImage, gradImage, Size(3, 3), 0.0);

    backModel.init(gradImage, ViBe::Config::getGradientConfig());

	ratioForeToFull.clear();
	ratioForeToFull.push_back(0);
	maxRatioForSparse = 0.05;
	frameCount = 1;

	saveRects.clear();
	rectsNoUpdate.clear();
}

void VideoAnalyzer::proc(Mat& image)
{
	medianBlur(image, blurImage, 3);
	GaussianBlur(blurImage, blurImage, Size(3, 3), 0.0);
	cvtColor(blurImage, grayImage, CV_BGR2GRAY);
	//calcThresholdedGradient(grayImage, gradImage, 200);
	calcGradient(grayImage, gradImage);
	medianBlur(gradImage, gradImage, 3);
	GaussianBlur(gradImage, gradImage, Size(3, 3), 0.0);

	backModel.update(gradImage, gradForeImage, rectsNoUpdate);
	
	vector<vector<Point> > contours;
	vector<RectInfo> rects;
	// 找前景轮廓
	Mat tempForeImage = gradForeImage.clone();
	findContours(tempForeImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	// 筛除小的前景
	for (int i = 0; i < contours.size(); i++)
	{
		if (contourArea(contours[i]) < 100) 
			continue;
		Rect currRect = boundingRect(contours[i]);
		if (currRect.width < 15 || currRect.height < 15)
			continue;
		rects.push_back(RectInfo(currRect));
	}

	// 如果 saveRects 为空，则将当前帧找到的所有 rects 放到 saveRects 中
	if (saveRects.empty())
	{
		for (int i = 0; i < rects.size(); i++)
			saveRects.push_back(RectInfo(rects[i]));
	}
	// 否则
	else
	{
		for (int i = 0; i < rects.size(); i++)
			rects[i].matchCount = 0;
			
		// 如果 saveRects 中的元素和某个 rects 中的元素能够完美匹配 则修改 saveRects 中元素的 matchCount
		for (vector<RectInfo>::iterator itr = saveRects.begin(); itr != saveRects.end();)
		{
			bool match = false;
			for (int i = 0; i < rects.size(); i++)
			{
				if (rects[i].matchCount)
					continue;

				Rect intersectRect = itr->rect & rects[i].rect;
				Rect unionRect = itr->rect | rects[i].rect;
				if (intersectRect.area() > 0.95 * unionRect.area())
				{
					rects[i].matchCount = 1;
					itr->rect = rects[i].rect;
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
				if (itr->missCount > 10)
					itr = saveRects.erase(itr);
				else
					++itr;
			}
			else
			{
				if (itr->matchCount > 20)
					itr = saveRects.erase(itr);
				else
					++itr;
			}
		}

		// 未能和 saveRects 中任何元素匹配的 rects 中的元素直接放到 saveRects 中
		for (int i = 0; i < rects.size(); i++)
		{
			if (!rects[i].matchCount)
				saveRects.push_back(RectInfo(rects[i].rect));
		}
			
		// 检查 saveRects 中所有元素 matchCount 的值
		// 大于阈值且面积较大的矩形放到 rectsNoUpdate 中
		rectsNoUpdate.clear();
		for (int i = 0; i < saveRects.size(); i++)
		{
			if (saveRects[i].matchCount > 40 && saveRects[i].rect.area() > 250)
			{
#if VIDEO_SPLIT_CMPL_LOG
				cout << "stable rect: "
					 << "x = " << saveRects[i].rect.x << ", "
					 << "y = " << saveRects[i].rect.y << ", "
					 << "w = " << saveRects[i].rect.width << ", "
					 << "h = " << saveRects[i].rect.height << "\n";
#endif
				rectsNoUpdate.push_back(saveRects[i].rect);
			}
		}
	}

	int forePixSum = 0;
	for (int i = 0; i < rects.size(); i++)
		forePixSum += rects[i].rect.width * rects[i].rect.height;
	ratioForeToFull.push_back(double(forePixSum) / pixSum);
	frameCount++;

#if VIDEO_SPLIT_CMPL_SHOW
		for (int i = 0; i < rects.size(); i++)
			rectangle(blurImage, rects[i].rect, Scalar(255, 255, 0));

		for (int i = 0; i < rectsNoUpdate.size(); i++)
			rectangle(blurImage, rectsNoUpdate[i], Scalar(0, 255, 0));

		showArrayByVertBar("Ratio Fore To Full", ratioForeToFull, false, true);

		imshow("blur image", blurImage);
		imshow("grad image", gradImage);
		imshow("grad fore image", gradForeImage);
        //waitKey(0);
#endif
}

int VideoAnalyzer::findSplitPosition(int expectCount)
{
	if (expectCount <= 0)
		return 0;
	if (expectCount >= frameCount)
		return frameCount - 1;

	vector<unsigned char> isSparse(frameCount, 0);
	for (int i = 0; i < frameCount; i++)
	{
		if (ratioForeToFull[i] < maxRatioForSparse)
			isSparse[i] = 1;
	}

#if VIDEO_SPLIT_CMPL_SHOW
		showArrayByVertBar("Ratio Fore To Full", ratioForeToFull, false, true, true, 0, 1, true, 200);
		showArrayByVertBar("Is Sparse", isSparse, false, true, true, 0, 1, true, 100);
        //waitKey(0);
#endif

	vector<Segment<unsigned char> > isSparseSeg;
	findSegments(isSparse, isSparseSeg);
	int numOfSeg = isSparseSeg.size();
	bool wideSegExists = false;
	for (int i = 0; i < numOfSeg; i++)
	{
		if (isSparseSeg[i].length > frameCount * 0.1)
		{
			wideSegExists = true;
			break;
		}
	}
	if (!wideSegExists)
		return expectCount;

	vector<double> testRatio;
	vector<int> index;	
	testRatio.reserve(numOfSeg);
	index.reserve(numOfSeg);
	for (int i = 0; i < numOfSeg; i++)
	{
		if (isSparseSeg[i].data)
		{
			testRatio.push_back(double(isSparseSeg[i].end - isSparseSeg[i].begin) /
				                double(abs((isSparseSeg[i].end + isSparseSeg[i].begin) / 2 - expectCount)));
			index.push_back(i);
		}
	}
	if (testRatio.size() < 1)
		return expectCount;

	int maxIndex = -1;
	double maxRatio = -1;
	for (int i = 0; i < testRatio.size(); i++)
	{
		if (testRatio[i] > maxRatio)
		{
			maxIndex = i;
			maxRatio = testRatio[i];
		}
	}

	return (isSparseSeg[index[maxIndex]].begin + isSparseSeg[index[maxIndex]].end) / 2;
}

void VideoAnalyzer::release(void)
{

}

}

static string getIntString(int val)
{
    stringstream strm;
    strm << val;
    return strm.str();
}

static string getDoubleString(double val)
{
    stringstream strm;
    strm << val;
    return strm.str();
}

namespace zpv
{

bool findSplitPositions(const string& videoPath, const double expectLengthInSecond,
    double& videoLengthInSecond, vector<double>& segmentLengthInSecond,
    vector<pair<int, int> >& splitBegAndEnd)
{
	videoLengthInSecond = 0;
    segmentLengthInSecond.clear();
    splitBegAndEnd.clear();

    VideoCapture videoCap;
	Mat frame, image;

#if VIDEO_SPLIT_CMPL_LOG
	string videoName = videoPath;
	int posSlash = videoName.find_last_of("\\/");
	if (posSlash != string::npos) 
		videoName = videoName.substr(posSlash + 1);
	createDirectory("result");
	stringstream logFileName;
	logFileName << "result/LogForVideoSplitter[" << videoName << "].txt";
	fstream logFile;
	logFile.open(logFileName.str().c_str(), ios::out);
#endif
	if (!videoCap.open(videoPath))
	{
#if VIDEO_SPLIT_CMPL_LOG
		stringstream message;
		message << "ERROR in function findSplitPositions(), "
			    << "cannot open file " << videoPath;
		logFile << message.str() << "\n";
		logFile.close();
		cerr << message.str() << "\n";
#endif
		THROW_EXCEPT("cannot open file " + videoPath);
	}
	double videoFrameCount = videoCap.get(CV_CAP_PROP_FRAME_COUNT);
	double videoFrameRate = videoCap.get(CV_CAP_PROP_FPS);
	if (videoFrameCount < 1.0)
	{
#if VIDEO_SPLIT_CMPL_LOG
		string message;
		message = "ERROR in function findSplitPositions(), "
				  "video frame count is invalid";
		logFile << message << "\n";
		logFile.close();
		cerr << message << "\n";
#endif
		THROW_EXCEPT("video frame count is " + getIntString(videoFrameCount) + ", invalid");
	}
	if (videoFrameRate < 1.0)
	{
#if VIDEO_SPLIT_CMPL_LOG
		string message;
		message = "ERROR in function findSplitPositions(), "
				  "video frame rate is invalid";
		logFile << message << "\n";
		logFile.close();
		cerr << message << "\n";
#endif
		THROW_EXCEPT("video frame rate is " + getDoubleString(videoFrameRate) + ", invalid");
	}
	videoLengthInSecond = videoFrameCount / videoFrameRate;
#if VIDEO_SPLIT_CMPL_LOG
	stringstream infoStr;
	infoStr << "Display video information:" << "\n";
	infoStr << fixed;
	infoStr << "frame count = " << setprecision(2) << videoFrameCount << ", "
		    << "frame rate = " << setprecision(2) << videoFrameRate << ", "
		    << "length in second = " << setprecision(2) << videoLengthInSecond << "\n";
	cout << infoStr.str();
	logFile << infoStr.str();
#endif
	double marginInSecond = 15;
	int width = 320;
	int height = 240;

	double splitUnitInSecond = expectLengthInSecond;
    if (splitUnitInSecond < 300)
	{
		splitUnitInSecond = 300;
#if VIDEO_SPLIT_CMPL_LOG
	    cout << "WARNING in function findSplitPositions(), splitUnitInSecond is set to 300" << "\n";
	    logFile << "WARNING in function findSplitPositions(), splitUnitInSecond is set to 300" << "\n";
#endif
	}
	int initNumOfSeg = int(videoLengthInSecond) / int(splitUnitInSecond);
	int numOfSeg;
	if (initNumOfSeg == 0)
		numOfSeg = 1;
	else if (videoLengthInSecond - initNumOfSeg * splitUnitInSecond > 0.65 * splitUnitInSecond)
		numOfSeg = initNumOfSeg + 1;
	else
		numOfSeg = initNumOfSeg;
	if (numOfSeg < 2)
	{
		segmentLengthInSecond.clear();
		segmentLengthInSecond.push_back(videoLengthInSecond);
		splitBegAndEnd.clear();
		splitBegAndEnd.push_back(make_pair(0, videoFrameCount - 1));
#if VIDEO_SPLIT_CMPL_LOG
		cout << fixed;
		cout << "WARINING in function findSplitPositions(), can only get one segment" << "\n";
		cout << "begin frame count = " << setw(10) << 0 << ", "
			 << "end frame count = " << setw(10) << int(videoFrameCount - 1) << ", "
			 << "time = " << setw(10) << setprecision(4) << videoFrameCount / videoFrameRate
			 << " sec" << "\n";
		logFile << fixed;
		logFile << "WARINING in function findSplitPositions(), can only get one segment" << "\n";
		logFile << "begin frame count = " << setw(10) << 0 << ", "
			    << "end frame count = " << setw(10) << int(videoFrameCount - 1) << ", "
				<< "time = " << setw(10) << setprecision(4) << videoFrameCount / videoFrameRate
				<< " sec"<< "\n";
		logFile << "Log ends" << "\n";
		logFile.close();
#endif
		return true;
	}
#if VIDEO_SPLIT_CMPL_LOG
	cout << "num of segments = " << numOfSeg << "\n";
#endif

	segmentLengthInSecond.clear();
	splitBegAndEnd.clear();
	vector<int> splitFramePos;
	splitFramePos.push_back(0);
	for (int j = 1; j < numOfSeg; j++)
	{
		int center, begInc, endExc;
		center = splitUnitInSecond * j * videoFrameRate;
		begInc = (splitUnitInSecond * j - marginInSecond) * videoFrameRate;
		endExc = (splitUnitInSecond * j + marginInSecond) * videoFrameRate;

		if (!videoCap.set(CV_CAP_PROP_POS_FRAMES, begInc))
		{
#if VIDEO_SPLIT_CMPL_LOG
			string message = "ERROR in function findSplitPositions(), "
					         "cannot seek designated frame";
			logFile << message << "\n";
			logFile.close();
			cerr << message << "\n";
#endif
			THROW_EXCEPT("cannot seek designated frame, frame count " + getIntString(begInc));
		}
		if (!videoCap.read(frame))
		{
#if VIDEO_SPLIT_CMPL_LOG
			string message = "ERROR in function findSplitPositions(), "
					         "cannot read designated frame";
			logFile << message << "\n";
			logFile.close();
			cerr << message << "\n";
#endif
			THROW_EXCEPT("cannot read designated frame, frame count " + getIntString(begInc));
		}
		resize(frame, image, Size(width, height), INTER_LINEAR);

		try
		{
			zvs::VideoAnalyzer analyzer;
			analyzer.init(image);		

			for (int i = 1; i < endExc - begInc; i++)
			{
				if (!videoCap.read(frame))
				{
#if VIDEO_SPLIT_CMPL_LOG
					string message = "ERROR in function findSplitPositions(), "
							         "cannot read designated frame";
					logFile << message << "\n";
					logFile.close();
					cerr << message << "\n";
#endif
					THROW_EXCEPT("cannot read designated frame, frame count " + getIntString(begInc + i));
				}
				resize(frame, image, Size(width, height), INTER_LINEAR);
				analyzer.proc(image);
			}
			int cutPosition = analyzer.findSplitPosition((endExc - begInc) / 2);
#if VIDEO_SPLIT_CMPL_LOG
			cout << "split segment " << j - 1 << " and segment " << j << ": ";
			cout << "expected center frame count = " << center << ", "
				 << "allowed begInc frame count = " << begInc << ", "
				 << "allowed endExc frame count = " << endExc << ", "
				 << "real cut position = " << begInc + cutPosition << "\n";
#endif
			splitFramePos.push_back(begInc + cutPosition);

#if VIDEO_SPLIT_CMPL_SHOW
				VideoCapture frameExtractor;
				Mat extractFrame;
				frameExtractor.open(videoPath);
				frameExtractor.set(CV_CAP_PROP_POS_FRAMES, begInc + cutPosition);
				frameExtractor.read(extractFrame);	
				stringstream imageName;
				imageName << "split frame " << begInc + cutPosition;
				imshow(imageName.str(), extractFrame);
				waitKey(0);
				frameExtractor.release();
#endif	
		}
		catch (const exception& s)
		{
#if VIDEO_SPLIT_CMPL_LOG
            logFile << s.what() << "\n";
			logFile.close();
            cerr << s.what() << "\n";
#endif
            THROW_EXCEPT(s.what());
		}
	}
	videoCap.release();

	for (int i = 1; i < splitFramePos.size(); i++)
	{
		segmentLengthInSecond.push_back(double(splitFramePos[i] - splitFramePos[i - 1]) / videoFrameRate);
		splitBegAndEnd.push_back(make_pair(splitFramePos[i - 1], splitFramePos[i] - 1));
	}
	segmentLengthInSecond.push_back(double(videoFrameCount - splitFramePos[splitFramePos.size() - 1]) / videoFrameRate);
	splitBegAndEnd.push_back(make_pair(splitFramePos[splitFramePos.size() - 1], videoFrameCount - 1));

#if VIDEO_SPLIT_CMPL_LOG
	stringstream logStream;
	logStream << fixed;
	logStream << "Function findSplitPosition() successfully run, "
		      << splitFramePos.size() << " video segment(s) obtained" << "\n";
	for (int i = 0; i < segmentLengthInSecond.size(); i++)
	{
		logStream << "segment = " << setw(5) << i << ": "
				  << "begin frame count = " << setw(10) << splitBegAndEnd[i].first << ", "
				  << "end frame count = "   << setw(10) << splitBegAndEnd[i].second << ", "
				  << "time = "  << setw(10) << setprecision(4) << segmentLengthInSecond[i]
				  << " sec" << "\n";
	}
	cout << logStream.str();
	logFile << logStream.str();
	logFile << "Log ends" << "\n";
	logFile.close();
#endif

	return true;
}

bool findSplitPositions(const string& videoPath, const int expectSegNum,
    double& videoLengthInSecond, vector<double>& segmentLengthInSecond,
    vector<pair<int, int> >& splitBegAndEnd)
{
	videoLengthInSecond = 0;
    segmentLengthInSecond.clear();
    splitBegAndEnd.clear();

    VideoCapture videoCap;
	Mat frame, image;

#if VIDEO_SPLIT_CMPL_LOG
	string videoName = videoPath;
	int posSlash = videoName.find_last_of("\\/");
	if (posSlash != string::npos) 
		videoName = videoName.substr(posSlash + 1);
	createDirectory("result");
	stringstream logFileName;
	logFileName << "result/LogForVideoSplitter[" << videoName << "].txt";
	fstream logFile;
	logFile.open(logFileName.str().c_str(), ios::out);
#endif
	if (!videoCap.open(videoPath))
	{
#if VIDEO_SPLIT_CMPL_LOG
		stringstream message;
		message << "ERROR in function findSplitPositions(), "
			    << "cannot open file " << videoPath;
		logFile << message.str() << "\n";
		logFile.close();
		cerr << message.str() << "\n";
#endif
		THROW_EXCEPT("cannot open file " + videoPath);
	}
	double videoFrameCount = videoCap.get(CV_CAP_PROP_FRAME_COUNT);
	double videoFrameRate = videoCap.get(CV_CAP_PROP_FPS);
	if (videoFrameCount < 1.0)
	{
#if VIDEO_SPLIT_CMPL_LOG
		string message;
		message = "ERROR in function findSplitPositions(), "
				  "video frame count is invalid";
		logFile << message << "\n";
		logFile.close();
		cerr << message << "\n";
#endif
		THROW_EXCEPT("video frame count is " + getIntString(videoFrameCount) + ", invalid");
	}
	if (videoFrameRate < 1.0)
	{
#if VIDEO_SPLIT_CMPL_LOG
		string message;
		message = "ERROR in function findSplitPositions(), "
				  "video frame rate is invalid";
		logFile << message << "\n";
		logFile.close();
		cerr << message << "\n";
#endif
		THROW_EXCEPT("video frame rate is " + getDoubleString(videoFrameRate) + ", invalid");
	}
	videoLengthInSecond = videoFrameCount / videoFrameRate;
#if VIDEO_SPLIT_CMPL_LOG
	stringstream infoStr;
	infoStr << "Display video information:" << "\n";
	infoStr << fixed;
	infoStr << "frame count = " << setprecision(2) << videoFrameCount << ", "
		    << "frame rate = " << setprecision(2) << videoFrameRate << ", "
		    << "length in second = " << setprecision(2) << videoLengthInSecond << "\n";
	cout << infoStr.str();
	logFile << infoStr.str();
#endif
	double marginInSecond = 15;
	int width = 320;
	int height = 240;

    double splitUnitInSecond = videoLengthInSecond / expectSegNum;
    if (splitUnitInSecond < 300)
	{
		splitUnitInSecond = 300;
#if VIDEO_SPLIT_CMPL_LOG
	    cout << "WARNING in function findSplitPositions(), splitUnitInSecond is set to 300" << "\n";
	    logFile << "WARNING in function findSplitPositions(), splitUnitInSecond is set to 300" << "\n";
#endif
	}
	int initNumOfSeg = int(videoLengthInSecond) / int(splitUnitInSecond);
	int numOfSeg;
	if (initNumOfSeg == 0)
		numOfSeg = 1;
	else if (videoLengthInSecond - initNumOfSeg * splitUnitInSecond > 0.65 * splitUnitInSecond)
		numOfSeg = initNumOfSeg + 1;
	else
		numOfSeg = initNumOfSeg;
	if (numOfSeg < 2)
	{
		segmentLengthInSecond.clear();
		segmentLengthInSecond.push_back(videoLengthInSecond);
		splitBegAndEnd.clear();
		splitBegAndEnd.push_back(make_pair(0, videoFrameCount - 1));
#if VIDEO_SPLIT_CMPL_LOG
		cout << fixed;
		cout << "WARINING in function findSplitPositions(), can only get one segment" << "\n";
		cout << "begin frame count = " << setw(10) << 0 << ", "
			 << "end frame count = " << setw(10) << int(videoFrameCount - 1) << ", "
			 << "time = " << setw(10) << setprecision(4) << videoFrameCount / videoFrameRate
			 << " sec" << "\n";
		logFile << fixed;
		logFile << "WARINING in function findSplitPositions(), can only get one segment" << "\n";
		logFile << "begin frame count = " << setw(10) << 0 << ", "
			    << "end frame count = " << setw(10) << int(videoFrameCount - 1) << ", "
				<< "time = " << setw(10) << setprecision(4) << videoFrameCount / videoFrameRate
				<< " sec"<< "\n";
		logFile << "Log ends" << "\n";
		logFile.close();
#endif
		return true;
	}
#if VIDEO_SPLIT_CMPL_LOG
	cout << "num of segments = " << numOfSeg << "\n";
#endif

	segmentLengthInSecond.clear();
	splitBegAndEnd.clear();
	vector<int> splitFramePos;
	splitFramePos.push_back(0);
	for (int j = 1; j < numOfSeg; j++)
	{
		int center, begInc, endExc;
		center = splitUnitInSecond * j * videoFrameRate;
		begInc = (splitUnitInSecond * j - marginInSecond) * videoFrameRate;
		endExc = (splitUnitInSecond * j + marginInSecond) * videoFrameRate;

		if (!videoCap.set(CV_CAP_PROP_POS_FRAMES, begInc))
		{
#if VIDEO_SPLIT_CMPL_LOG
			string message = "ERROR in function findSplitPositions(), "
					         "cannot seek designated frame";
			logFile << message << "\n";
			logFile.close();
			cerr << message << "\n";
#endif
			THROW_EXCEPT("cannot seek designated frame, frame count " + getIntString(begInc));
		}
		if (!videoCap.read(frame))
		{
#if VIDEO_SPLIT_CMPL_LOG
			string message = "ERROR in function findSplitPositions(), "
					         "cannot read designated frame";
			logFile << message << "\n";
			logFile.close();
			cerr << message << "\n";
#endif
			THROW_EXCEPT("cannot read designated frame, frame count " + getIntString(begInc));
		}
		resize(frame, image, Size(width, height), INTER_LINEAR);

		try
		{
			zvs::VideoAnalyzer analyzer;
			analyzer.init(image);		

			for (int i = 1; i < endExc - begInc; i++)
			{
				if (!videoCap.read(frame))
				{
#if VIDEO_SPLIT_CMPL_LOG
					string message = "ERROR in function findSplitPositions(), "
							         "cannot read designated frame";
					logFile << message << "\n";
					logFile.close();
					cerr << message << "\n";
#endif
					THROW_EXCEPT("cannot read designated frame, frame count " + getIntString(begInc + i));
				}
				resize(frame, image, Size(width, height), INTER_LINEAR);
				analyzer.proc(image);
			}
			int cutPosition = analyzer.findSplitPosition((endExc - begInc) / 2);
#if VIDEO_SPLIT_CMPL_LOG
			cout << "split segment " << j - 1 << " and segment " << j << ": ";
			cout << "expected center frame count = " << center << ", "
				 << "allowed begInc frame count = " << begInc << ", "
				 << "allowed endExc frame count = " << endExc << ", "
				 << "real cut position = " << begInc + cutPosition << "\n";
#endif
			splitFramePos.push_back(begInc + cutPosition);

#if VIDEO_SPLIT_CMPL_SHOW
				VideoCapture frameExtractor;
				Mat extractFrame;
				frameExtractor.open(videoPath);
				frameExtractor.set(CV_CAP_PROP_POS_FRAMES, begInc + cutPosition);
				frameExtractor.read(extractFrame);	
				stringstream imageName;
				imageName << "split frame " << begInc + cutPosition;
				imshow(imageName.str(), extractFrame);
				waitKey(0);
				frameExtractor.release();
#endif	
		}
		catch (const exception& s)
		{
#if VIDEO_SPLIT_CMPL_LOG
            logFile << s.what() << "\n";
			logFile.close();
            cerr << s.what() << "\n";
#endif
            THROW_EXCEPT(s.what());
		}
	}
	videoCap.release();

	for (int i = 1; i < splitFramePos.size(); i++)
	{
		segmentLengthInSecond.push_back(double(splitFramePos[i] - splitFramePos[i - 1]) / videoFrameRate);
		splitBegAndEnd.push_back(make_pair(splitFramePos[i - 1], splitFramePos[i] - 1));
	}
	segmentLengthInSecond.push_back(double(videoFrameCount - splitFramePos[splitFramePos.size() - 1]) / videoFrameRate);
	splitBegAndEnd.push_back(make_pair(splitFramePos[splitFramePos.size() - 1], videoFrameCount - 1));

#if VIDEO_SPLIT_CMPL_LOG
	stringstream logStream;
	logStream << fixed;
	logStream << "Function findSplitPosition() successfully run, "
		      << splitFramePos.size() << " video segment(s) obtained" << "\n";
	for (int i = 0; i < segmentLengthInSecond.size(); i++)
	{
		logStream << "segment = " << setw(5) << i << ": "
				  << "begin frame count = " << setw(10) << splitBegAndEnd[i].first << ", "
				  << "end frame count = "   << setw(10) << splitBegAndEnd[i].second << ", "
				  << "time = "  << setw(10) << setprecision(4) << segmentLengthInSecond[i]
				  << " sec" << "\n";
	}
	cout << logStream.str();
	logFile << logStream.str();
	logFile << "Log ends" << "\n";
	logFile.close();
#endif

	return true;
}

}
