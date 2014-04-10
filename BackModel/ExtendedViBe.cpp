#include <cmath>
#include <fstream>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ExtendedViBe.h"
#include "Exception.h"

using namespace std;
using namespace cv;

const static int adjPositions[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

namespace zsfo
{

void ViBe::init(const Mat& image, const Config& config)
{
    if (image.cols <= 0 || image.rows <= 0 || image.type() != CV_8UC3 && image.type() != CV_8UC1)
		THROW_EXCEPT("unsupported image format");

	imageWidth = image.cols;
	imageHeight = image.rows;
    imageRect = Rect(0, 0, imageWidth, imageHeight);
	imageChannels = image.channels();
    imageType = image.type();

    numOfSamples = config.numOfSamples;
    minMatchDist = config.minMatchDist;
    minNumOfMatchCount = config.minNumOfMatchCount;
    subSampleInterval = config.subSampleInterval;
    //printf("display ViBe config for %s:\n", config.label.c_str());
	//printf("  numOfSamples = %d\n", numOfSamples);
	//printf("  minMatchDist = %d\n", minMatchDist);
	//printf("  minNumOfMatchCount = %d\n", minNumOfMatchCount);
	//printf("  subSampleInterval = %d\n", subSampleInterval);
	//printf("\n");

	// 初始化随机数
	rndReplaceCurr.init(imageWidth * imageHeight, 0, subSampleInterval, getTickCount());
	rndIndexCurr.init(imageWidth * imageHeight, 0, numOfSamples, getTickCount() / 2);
	rndReplaceAdj.init(imageWidth * imageHeight, 0, subSampleInterval, getTickCount() / 3);
	rndPositionAdj.init(imageWidth * imageHeight, 0, 8, getTickCount() / 4);
	rndIndexAdj.init(imageWidth * imageHeight, 0, numOfSamples, getTickCount() / 5); 

	// 不更新区域图
	noUpdateImage = Mat::zeros(imageHeight, imageWidth, CV_8UC1);
	rowNoUpdate.resize(imageHeight, 0);
	for (int i = 0; i < imageHeight; i++)
		rowNoUpdate[i] = noUpdateImage.ptr<unsigned char>(i);
    ptrNoUpdate = &rowNoUpdate[0];
	
	// 分配保存样本空间 标记行首地址
    samples = Mat::zeros(imageWidth * imageHeight * imageChannels * numOfSamples, 1, CV_8UC1);
    rowSamples.resize(imageHeight, 0);
	for (int i = 0; i < imageHeight; i++)
        rowSamples[i] = samples.data + imageWidth * numOfSamples * imageChannels * i;
    ptrSamples = &rowSamples[0];

	// 填充背景样本和背景图片
	if (imageChannels == 3)
		fill8UC3(image);
	else if (imageChannels == 1)
		fill8UC1(image);
}

void ViBe::refill(const Mat& image)
{
	if (image.type() != imageType)
		THROW_EXCEPT("image.type() != imageType");
	if (image.cols != imageWidth || image.rows != imageHeight)
		THROW_EXCEPT("image size does not match");

	if (imageChannels == 3)
		fill8UC3(image);
	else if (imageChannels == 1)
		fill8UC1(image);
}

void ViBe::fill8UC3(const Mat& image)
{
	RandUniformInt rndInit;
	rndInit.init(imageWidth * imageHeight * numOfSamples, 0, 8, getTickCount());
	// 输入图片的行首地址
	const unsigned char** ptrImage = new const unsigned char* [imageHeight];
	for (int i = 0; i < imageHeight; i++)
	{
		ptrImage[i] = image.ptr<unsigned char>(i);
	}
	for (int i = 1; i < imageHeight - 1; i++)
	{
		for (int j = 1; j < imageWidth - 1; j++)
		{
			for (int k = 0; k < numOfSamples; k++)
			{
				int index = rndInit.getNext();
				memcpy(ptrSamples[i] + (j * numOfSamples + k) * 3, 
						ptrImage[i + adjPositions[index][0]] + (j + adjPositions[index][1]) * 3,
						sizeof(unsigned char) * 3);
			}
		} 
	}
	delete [] ptrImage;
}

void ViBe::fill8UC1(const Mat& image)
{
	RandUniformInt rndInit;
	rndInit.init(imageWidth * imageHeight * numOfSamples, 0, 8, getTickCount());
	// 输入图片的行首地址
	const unsigned char** ptrImage = new const unsigned char* [imageHeight];
	for (int i = 0; i < imageHeight; i++)
	{
		ptrImage[i] = image.ptr<unsigned char>(i);
	}
	for (int i = 1; i < imageHeight - 1; i++)
	{
		for (int j = 1; j < imageWidth - 1; j++)
		{
			for (int k = 0; k < numOfSamples; k++)
			{
				int index = rndInit.getNext();
				memcpy(ptrSamples[i] + (j * numOfSamples + k), 
						ptrImage[i + adjPositions[index][0]] + (j + adjPositions[index][1]),
						sizeof(unsigned char));
			}
		} 
	}
	delete [] ptrImage;
}

void ViBe::update(const Mat& image, Mat& foreImage, const vector<Rect>& rectsNoUpdate)
{	
	if (image.type() != imageType)
		THROW_EXCEPT("image.type() != imageType");

	if (image.cols != imageWidth || image.rows != imageHeight)
		THROW_EXCEPT("image size does not match");

	noUpdateImage.setTo(0);
	if (!rectsNoUpdate.empty())
	{
        for (int i = 0; i < rectsNoUpdate.size(); i++)
		{
			Mat noUpdateMatROI = noUpdateImage(rectsNoUpdate[i] & imageRect);
			noUpdateMatROI.setTo(255);
		}
	}
    foreImage.create(imageHeight, imageWidth, CV_8UC1);

	if (imageChannels == 3)
        proc8UC3(image, foreImage);
	else if (imageChannels == 1)
        proc8UC1(image, foreImage);
}

void ViBe::proc8UC3(const Mat& image, Mat& foreImage)
{
	const unsigned char** ptrImage = new const unsigned char* [imageHeight];
	for (int i = 0; i < imageHeight; i++)
	{
		ptrImage[i] = image.ptr<unsigned char>(i);
	}

	for (int i = 1; i < imageHeight - 1; i++)
	{
		unsigned char* ptrFore = foreImage.ptr<unsigned char>(i);
        for (int j = 1; j < imageWidth - 1; j++)
		{
			// 统计当前像素能够和多少个已存储的样本匹配
			int matchCount = 0;
			const unsigned char* ptrInput = ptrImage[i] + j * 3;
			unsigned char* ptrStore = ptrSamples[i] + j * numOfSamples * 3;
			for (int k = 0; k < numOfSamples && matchCount < minNumOfMatchCount; k++)
			{
				int dist = abs(int(ptrInput[0]) - int(ptrStore[k * 3])) +
						abs(int(ptrInput[1]) - int(ptrStore[k * 3 + 1])) +
						abs(int(ptrInput[2]) - int(ptrStore[k * 3 + 2]));
				if (dist < minMatchDist)
					matchCount++;
			}

			// 是前景
			if (matchCount < minNumOfMatchCount)
			{
				ptrFore[j] = 255;
				continue;
			}

			// 是背景
			ptrFore[j] = 0;

			if (ptrNoUpdate[i][j])
				continue;

			// 更新当前像素的存储样本
			if (rndReplaceCurr.getNext() == 0)
			{
				memcpy(ptrStore + rndIndexCurr.getNext() * 3, 
						ptrInput, sizeof(unsigned char) * 3);
			}

			// 更新邻域像素的存储样本
			if (rndReplaceAdj.getNext() == 0)
			{
				int posAdj = rndPositionAdj.getNext();
				memcpy(ptrSamples[i + adjPositions[posAdj][0]] + 
						((j + adjPositions[posAdj][1]) * numOfSamples + rndIndexAdj.getNext()) * 3,
						ptrInput, sizeof(unsigned char) * 3);
			}
		}
	}
	delete [] ptrImage;
}

void ViBe::proc8UC1(const Mat& image, Mat& foreImage)
{
	const unsigned char** ptrImage = new const unsigned char* [imageHeight];
	for (int i = 0; i < imageHeight; i++)
	{
		ptrImage[i] = image.ptr<unsigned char>(i);
	}

	for (int i = 1; i < imageHeight - 1; i++)
	{
		unsigned char* ptrFore = foreImage.ptr<unsigned char>(i);
        for (int j = 1; j < imageWidth - 1; j++)
		{
			// 统计当前像素能够和多少个已存储的样本匹配
			int matchCount = 0;
			const unsigned char* ptrInput = ptrImage[i] + j;
			unsigned char* ptrStore = ptrSamples[i] + j * numOfSamples;
			for (int k = 0; k < numOfSamples && matchCount < minNumOfMatchCount; k++)
			{
				int dist = abs(int(ptrInput[0]) - int(ptrStore[k]));
				if (dist < minMatchDist)
					matchCount++;
			}

			// 是前景
			if (matchCount < minNumOfMatchCount)
			{
				ptrFore[j] = 255;
				continue;
			}

			// 是背景
			ptrFore[j] = 0;

			if (ptrNoUpdate[i][j])
				continue;

			// 更新当前像素的存储样本
			if (rndReplaceCurr.getNext() == 0)
			{
				memcpy(ptrStore + rndIndexCurr.getNext(), 
						ptrInput, sizeof(unsigned char));
			}

			// 更新邻域像素的存储样本
			if (rndReplaceAdj.getNext() == 0)
			{
				int posAdj = rndPositionAdj.getNext();
				memcpy(ptrSamples[i + adjPositions[posAdj][0]] + 
						((j + adjPositions[posAdj][1]) * numOfSamples + rndIndexAdj.getNext()),
						ptrInput, sizeof(unsigned char));
			}
		}
	}
	delete [] ptrImage;
}

void ViBe::showSamples(int count)
{
	if (count <= 0 || count > numOfSamples)
		return;

	Mat* samplesMat = new Mat [count];
	if (imageChannels == 3)
	{
		
		for (int i = 0; i < count; i++)
		{
			samplesMat[i].create(imageHeight, imageWidth, CV_8UC3);
		}

		for (int i = 0; i < imageHeight; i++)
		{
			unsigned char** ptr = new unsigned char* [count];
			for (int k = 0; k < count; k++)
			{
				ptr[k] = samplesMat[k].ptr<unsigned char>(i);
			}
			for (int j = 0; j < imageWidth; j++)
			{			
				for (int k = 0; k < count; k++)
				{
					memcpy(ptr[k] + j * 3, 
						   ptrSamples[i] + (j * numOfSamples + k) * 3, 
						   sizeof(unsigned char) * 3);
				}			
			}
			delete [] ptr;
		}
	}
	else if (imageChannels == 1)
	{
		for (int i = 0; i < count; i++)
		{
			samplesMat[i].create(imageHeight, imageWidth, CV_8UC1);
		}

		for (int i = 0; i < imageHeight; i++)
		{
			unsigned char** ptr = new unsigned char* [count];
			for (int k = 0; k < count; k++)
			{
				ptr[k] = samplesMat[k].ptr<unsigned char>(i);
			}
			for (int j = 0; j < imageWidth; j++)
			{			
				for (int k = 0; k < count; k++)
				{
					memcpy(ptr[k] + j, 
						   ptrSamples[i] + (j * numOfSamples + k), 
						   sizeof(unsigned char));
				}			
			}
			delete [] ptr;
		}
	}

	for (int i = 0; i < count; i++)
	{
		char imageName[100];
		sprintf(imageName, "samples %d", i);
		imshow(imageName, samplesMat[i]);
	}
	delete [] samplesMat;
}

void ExtendedViBe::init(const Mat& image, const ExtendedConfig& config)
{
    ViBe::init(image, config);
	learnRate = config.learnRate;
	compLearnRate = 1.F - learnRate;
    //printf("display ExtendedViBe config for %s:\n", config.label.c_str());
	//printf("  learnRate = %.4f\n", learnRate);
	//printf("  compLearnRate = %.4f\n", compLearnRate);
	//printf("\n");
    if (imageChannels == 3)
		image.convertTo(backImage, CV_32FC3);
	else if (imageChannels == 1)
		image.convertTo(backImage, CV_32FC1);
}

void ExtendedViBe::update(const Mat& image, Mat& foregroundImage, Mat& backgroundImage, 
    const vector<Rect>& rectsNoUpdate)
{
    ViBe::update(image, foregroundImage, rectsNoUpdate);
    accumulateWeighted(image, backImage, learnRate, ~(foregroundImage | noUpdateImage));
    if (imageChannels == 3)
		backImage.convertTo(backgroundImage, CV_8UC3);
	else if (imageChannels == 1)
		backImage.convertTo(backgroundImage, CV_8UC1);
}

void ExtendedViBe::refill(const Mat& image)
{
    ViBe::refill(image);
    if (imageChannels == 3)
        image.convertTo(backImage, CV_32FC3);
    else if (imageChannels == 1)
        image.convertTo(backImage, CV_32FC1);
}

}