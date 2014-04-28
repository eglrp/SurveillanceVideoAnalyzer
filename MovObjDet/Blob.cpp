#include "BlobTrackerInternal.h"
#include "FilestreamScopeGuard.h"
#include "Exception.h"
#include "CompileControl.h"

using namespace std;
using namespace cv;
using namespace ztool;

namespace zsfo
{

Blob::Blob(const Ptr<SizeInfo>& sizesOrigAndNorm, 
    const Ptr<long long int>& time, const Ptr<int>& count, int blobID, const string& path)
    : ID(blobID),  
      configOutputInfo(new ConfigOutputInfo)
{
    initConfigParam(path);
    rectHistory = new BlobQuanHistory(sizesOrigAndNorm, time, count, blobID);
}

Blob::Blob(const Ptr<LineSegment>& crossLine, const Ptr<SizeInfo>& sizesOrigAndNorm, 
    const Ptr<Rect>& baseRectangle, const Ptr<long long int>& time, const Ptr<int>& count, 
    int blobID, int saveMode, const string& path)
    : ID(blobID),    
      configOutputInfo(new ConfigOutputInfo)
{	
    initConfigParam(path);
    rectHistory = new BlobQuanHistory(sizesOrigAndNorm, time, count, blobID);
    visualHistory = 
        (BlobVisualHistory*) new BlobCrossLineVisualHistory(crossLine, 
        sizesOrigAndNorm, baseRectangle, time, count, blobID, saveMode, path);
}

Blob::Blob(const Ptr<VirtualLoop>& recordLoop, const Ptr<SizeInfo>& sizesOrigAndNorm, 
    const Ptr<Rect>& baseRectangle, const Ptr<long long int>& time, const Ptr<int>& count, 
    int blobID, bool isTriBound, int saveMode, const string& path)
    : ID(blobID),
      configOutputInfo(new ConfigOutputInfo)
{
    initConfigParam(path);
    rectHistory = new BlobQuanHistory(sizesOrigAndNorm, time, count, blobID);
    if (isTriBound)
    {
        visualHistory = 
            (BlobVisualHistory*) new BlobTriBoundVisualHistory(recordLoop, 
            sizesOrigAndNorm, baseRectangle, time, count, blobID, saveMode, path);
    }
    else
    {
        visualHistory = 
            (BlobVisualHistory*) new BlobBottomBoundVisualHistory(recordLoop, 
            sizesOrigAndNorm, baseRectangle, time, count, blobID, saveMode, path);
    }
}

Blob::Blob(const Ptr<SizeInfo>& sizesOrigAndNorm, const Ptr<Rect>& baseRectangle, 
    const Ptr<long long int>& time, const Ptr<int>& count, 
    int blobID, int saveMode, int saveInterval, int numOfSaved, const string& path)
    : ID(blobID),
      configOutputInfo(new ConfigOutputInfo)
{
    initConfigParam(path);
    rectHistory = new BlobQuanHistory(sizesOrigAndNorm, time, count, blobID);
    visualHistory =
        (BlobVisualHistory*) new BlobMultiRecordVisualHistory(sizesOrigAndNorm, baseRectangle, time, count, blobID,
        saveMode, saveInterval, numOfSaved, path);
}

Blob::Blob(const Blob& blob, int blobID, const Rect& rect) :
    ID(blobID), matchRect(rect), isToBeDeleted(false),
    rectHistory(blob.rectHistory->createNew(blobID)),
    visualHistory(blob.visualHistory ? blob.visualHistory->createNew(blobID) : 0),
    configOutputInfo(blob.configOutputInfo)
{

}

Blob::~Blob(void)
{

}

Blob* Blob::createNew(int blobID, const cv::Rect& rect) const
{
    Blob* ptrBlob = new Blob(*this, blobID, rect);
    return ptrBlob;
}

void Blob::initConfigParam(const string& path)
{
	if (!path.empty())
    {
        fstream initFileStream;
        FileStreamScopeGuard<fstream> guard(initFileStream);
        initFileStream.open(path.c_str());
	    if (!initFileStream.is_open())
	    {
            THROW_EXCEPT("cannot open file " + path);
	    }
        char stringNotUsed[1024];
	    do
	    {
		    initFileStream >> stringNotUsed;
		    if (initFileStream.eof())
		    {
                THROW_EXCEPT("cannot find config params label [Blob] for Blob");
		    }
	    }
	    while(string(stringNotUsed) != string("[Blob]"));
	
	    initFileStream >> stringNotUsed >> stringNotUsed;
	    initFileStream >> configOutputInfo->minHistorySizeForOutput;
	    initFileStream >> stringNotUsed;
	    initFileStream >> configOutputInfo->runOutputHistory;
	    initFileStream >> stringNotUsed;
	    initFileStream >> configOutputInfo->runOutputVisualAndState;

	    initFileStream.close();
    }
    else
    {
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
        printf("BlobTracker is initialized with default param\n");
#endif
        configOutputInfo->minHistorySizeForOutput = 0;
        configOutputInfo->runOutputHistory = true;
        configOutputInfo->runOutputVisualAndState = true;
    }

#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
	printf("display blob config:\n");

	printf("  function output info:\n");
	printf("    configOutputInfo.minHistorySizeForOutput = %d\n", configOutputInfo->minHistorySizeForOutput);
	printf("    configOutputInfo.runOutputHistory = %s\n", configOutputInfo->runOutputHistory ? "true" : "false");
	printf("    configOutputInfo.runOutputVisualAndState = %s\n", configOutputInfo->runOutputVisualAndState ? "true" : "false");

	printf("\n");
#endif
}

void Blob::updateState(void)
{
    if (isToBeDeleted)
        return;
    rectHistory->pushRecord(matchRect, 0.0);
}

void Blob::updateState(OrigSceneProxy& origFrame, OrigForeProxy& foreImage)
{
    if (isToBeDeleted)
        return;
    rectHistory->pushRecord(matchRect, 0.0);
    if (visualHistory) visualHistory->updateHistory(origFrame, foreImage, matchRect);
}

void Blob::updateState(OrigSceneProxy& origFrame, OrigForeProxy& foreImage,
    const Mat& gradDiffImage, const Mat& lastGradDiffImage)
{
    if (isToBeDeleted)
        return;
    // 计算当前梯度差和上一帧梯度差的差值的均值，并进行记录
	Rect unionRect = matchRect | rectHistory->currRecord.rect;
	Mat lastGradDiffRegion = Mat(lastGradDiffImage, unionRect);
	Mat gradDiffRegion = Mat(gradDiffImage, unionRect);
	Scalar gradDiffMean = mean(abs(lastGradDiffRegion - gradDiffRegion));
    rectHistory->pushRecord(matchRect, gradDiffMean[0]);
    if (visualHistory) visualHistory->updateHistory(origFrame, foreImage, matchRect);
}

bool Blob::outputInfo(ObjectInfo& objectInfo, bool isFinal) const
{
	objectInfo.ID = ID;		
	objectInfo.currRect = rectHistory->currRecord.origRect;

	if (!isToBeDeleted && !isFinal)
	{
		objectInfo.isFinal = 0;
		objectInfo.hasHistory = 0;
		return true;
	}

	objectInfo.isFinal = 1;

	if (rectHistory->history.size() < configOutputInfo->minHistorySizeForOutput)
	{
		objectInfo.hasHistory = 0;
#if CMPL_WRITE_CONSOLE || CMPL_WRITE_NECESSARY_CONSOLE
		printf("Blob ID: %d History size too small, no image output\n", ID);
#endif
		return true;
	}

	if (configOutputInfo->runOutputHistory)
		rectHistory->outputHistory(objectInfo);

    if (configOutputInfo->runOutputVisualAndState)
        if (visualHistory) visualHistory->outputHistory(objectInfo);

	return true;
}

void Blob::drawBlob(Mat& normalImage, const Scalar& color) const
{
	if (!isToBeDeleted)
		rectHistory->drawRect(normalImage, color);
}

void Blob::drawHistory(Mat& normalImage, const Scalar& color) const
{
    rectHistory->drawCenterHistory(normalImage, color);
}

int Blob::getID(void) const
{
    return ID;
}

cv::Rect Blob::getCurrRect(void) const
{
    return matchRect;
}

bool Blob::getIsToBeDeleted(void) const
{
    return isToBeDeleted;
}

void Blob::setCurrRect(const Rect& rect) 
{
    matchRect = rect;
}

void Blob::setToBeDeleted(void) 
{
    isToBeDeleted = true;
}

int Blob::getHistoryLength(void) const
{
    if (rectHistory)
        return (int)rectHistory->history.size();
    else 
        return 0;
}

void Blob::getCenterHistory(vector<Point>& centerHistory) const
{
    centerHistory.clear();
    if (rectHistory)
        rectHistory->getCenterHistory(centerHistory);
}

bool Blob::doesTurnAround(void) const
{
    if (rectHistory) 
        return rectHistory->checkTurnAround(); 
    else 
        return false;
}

void Blob::printHistory(void) const
{
    rectHistory->displayHistory();
}

}