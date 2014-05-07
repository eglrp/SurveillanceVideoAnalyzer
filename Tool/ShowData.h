#pragma once

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace ztool
{

template<typename Type>
void showArrayByVertBar(const std::string& name, const std::vector<Type>& arr, bool showMean, bool showZero)
{
	if (arr.empty()) return;
    double minVal = arr[0];
	double maxVal = arr[0];
	double avg = 0;
	int arrLength = arr.size();
	for (int i = 0; i < arrLength; i++)
	{
		minVal = arr[i] < minVal ? arr[i] : minVal;
		maxVal = arr[i] > maxVal ? arr[i] : maxVal;
		avg += arr[i];
	}
	avg /= arrLength;	
	if (showZero)
	{
		maxVal = maxVal < 0 ? 0 : maxVal;
		minVal = minVal > 0 ? 0 : minVal;
	}
	if (maxVal == minVal)
	{
		if (maxVal > 0)
		{
			minVal = 0;
			avg = maxVal;
		}
		else if (maxVal < 0)
		{
			maxVal = 0;
			avg = minVal;
		}
		else
		{
			maxVal = 1;
			minVal = 0;
			avg = 0;
		}
	}
	
	int margin = 10;
	int barLength = 380;
	double scale = double(barLength) / (maxVal - minVal);
	cv::Mat image = cv::Mat::zeros(barLength + 2 * margin, arrLength + 2 * margin, CV_8UC3);
	if (!showZero)
	{
		if (!showMean)
		{
			for (int i = 0; i < arrLength; i++)
			{
				cv::Point begin = cv::Point(i + margin, barLength + 2 * margin);
				cv::Point end = cv::Point(i + margin, barLength + margin - scale * (arr[i] - minVal));
				cv::line(image, begin, end, cv::Scalar::all(255));
			}
		}
		else
		{
			for (int i = 0; i < arrLength; i++)
			{
				cv::Point begin = cv::Point(i + margin, barLength + 2 * margin);
				cv::Point end = cv::Point(i + margin, barLength + margin - scale * (arr[i] - minVal));
				if (arr[i] < avg)
					cv::line(image, begin, end, cv::Scalar(255, 255, 0));
				else
					cv::line(image, begin, end, cv::Scalar(0, 255, 255));
			}
			int meanPosition = barLength + margin - scale * (avg - minVal);
			cv::line(image, cv::Point(margin, meanPosition), cv::Point(arrLength + margin, meanPosition), cv::Scalar(0, 0, 255));
		}
	}
	else
	{
		int zeroPosition = barLength + margin - scale * (0 - minVal);
		if (!showMean)
		{
			for (int i = 0; i < arrLength; i++)
			{
				cv::Point begin = cv::Point(i + margin, zeroPosition);
				cv::Point end = cv::Point(i + margin, barLength + margin - scale * (arr[i] - minVal));
				cv::line(image, begin, end, cv::Scalar::all(255));
			}
			cv::line(image, cv::Point(margin, zeroPosition), cv::Point(arrLength + margin, zeroPosition), cv::Scalar(255, 255, 255));
		}
		else
		{
			for (int i = 0; i < arrLength; i++)
			{
				cv::Point begin = cv::Point(i + margin, zeroPosition);
				cv::Point end = cv::Point(i + margin, barLength + margin - scale * (arr[i] - minVal));
				if (arr[i] < avg)
					cv::line(image, begin, end, cv::Scalar(255, 255, 0));
				else
					cv::line(image, begin, end, cv::Scalar(0, 255, 255));
			}
			int meanPosition = barLength + margin - scale * (avg - minVal);
			cv::line(image, cv::Point(margin, zeroPosition), cv::Point(arrLength + margin, zeroPosition), cv::Scalar(255, 255, 255));
			cv::line(image, cv::Point(margin, meanPosition), cv::Point(arrLength + margin, meanPosition), cv::Scalar(0, 0, 255));
		}
	}
	cv::imshow(name, image);
}

template<typename Type>
void showArrayByHoriBar(const std::string& name, const std::vector<Type>& arr, bool showMean, bool showZero)
{
	if (arr.empty()) return;
    double minVal = arr[0];
	double maxVal = arr[0];
	double avg = 0;
	int arrLength = arr.size();
	for (int i = 0; i < arrLength; i++)
	{
		minVal = arr[i] < minVal ? arr[i] : minVal;
		maxVal = arr[i] > maxVal ? arr[i] : maxVal;
		avg += arr[i];
	}
	avg /= arrLength;	
	if (showZero)
	{
		maxVal = maxVal < 0 ? 0 : maxVal;
		minVal = minVal > 0 ? 0 : minVal;
	}
	if (maxVal == minVal)
	{
		if (maxVal > 0)
		{
			minVal = 0;
			avg = maxVal;
		}
		else if (maxVal < 0)
		{
			maxVal = 0;
			avg = minVal;
		}
		else
		{
			maxVal = 1;
			minVal = 0;
			avg = 0;
		}
	}
	
	int margin = 10;
	int barLength = 380;
	double scale = double(barLength) / (maxVal - minVal);
	cv::Mat image = cv::Mat::zeros(arrLength + 2 * margin, barLength + 2 * margin, CV_8UC3);
	if (!showZero)
	{
		if (!showMean)
		{
			for (int i = 0; i < arrLength; i++)
			{
				cv::Point begin = cv::Point(0, i + margin);
				cv::Point end = cv::Point(scale * (arr[i] - minVal) + margin, i + margin);
				cv::line(image, begin, end, cv::Scalar::all(255));
			}
		}
		else
		{
			for (int i = 0; i < arrLength; i++)
			{
				cv::Point begin = cv::Point(0, i + margin);
				cv::Point end = cv::Point(scale * (arr[i] - minVal), i + margin);
				if (arr[i] < avg)
					cv::line(image, begin, end, cv::Scalar(255, 255, 0));
				else
					cv::line(image, begin, end, cv::Scalar(0, 255, 255));
			}
			int meanPosition = scale * (avg - minVal) + margin;
			cv::line(image, cv::Point(meanPosition, margin), cv::Point(meanPosition, arrLength + margin), cv::Scalar(0, 0, 255));
		}
	}
	else
	{
		int zeroPosition = scale * (0 - minVal) + margin;
		if (!showMean)
		{
			for (int i = 0; i < arrLength; i++)
			{
				cv::Point begin = cv::Point(zeroPosition, i + margin);
				cv::Point end = cv::Point(scale * (arr[i] - minVal) + margin, i + margin);
				cv::line(image, begin, end, cv::Scalar::all(255));
			}
			cv::line(image, cv::Point(zeroPosition, margin), cv::Point(zeroPosition, arrLength + margin), cv::Scalar(255, 255, 255));
		}
		else
		{
			for (int i = 0; i < arrLength; i++)
			{
				cv::Point begin = cv::Point(zeroPosition, i + margin);
				cv::Point end = cv::Point(scale * (arr[i] - minVal) + margin, i + margin);
				if (arr[i] < avg)
					cv::line(image, begin, end, cv::Scalar(255, 255, 0));
				else
					cv::line(image, begin, end, cv::Scalar(0, 255, 255));
			}
			int meanPosition = scale * (avg - minVal) + margin;
			cv::line(image, cv::Point(zeroPosition, margin), cv::Point(zeroPosition, arrLength + margin), cv::Scalar(255, 255, 255));
			cv::line(image, cv::Point(meanPosition, margin), cv::Point(meanPosition, arrLength + margin), cv::Scalar(0, 0, 255));
		}
	}
	cv::imshow(name, image);
}

template<typename Type>
void showArrayByVertBar(const std::string& name, const std::vector<Type>& arr, bool showMean, bool showZero, 
	bool setRange, double initMin, double initMax, bool setBarLength, double initBarLength)
{
	double minVal = arr[0];
	double maxVal = arr[0];
	double avg = 0;
	int arrLength = arr.size();
	for (int i = 0; i < arrLength; i++)
	{
		minVal = arr[i] < minVal ? arr[i] : minVal;
		maxVal = arr[i] > maxVal ? arr[i] : maxVal;
		avg += arr[i];
	}
	avg /= arrLength;	
	if (setRange && initMin < initMax)
	{
		minVal = initMin < minVal ? initMin : minVal;
		maxVal = initMax > maxVal ? initMax : maxVal;
	}
	if (showZero)
	{
		maxVal = maxVal < 0 ? 0 : maxVal;
		minVal = minVal > 0 ? 0 : minVal;
	}
	if (maxVal == minVal)
	{
		if (maxVal > 0)
		{
			minVal = 0;
			avg = maxVal;
		}
		else if (maxVal < 0)
		{
			maxVal = 0;
			avg = minVal;
		}
		else
		{
			maxVal = 1;
			minVal = 0;
			avg = 0;
		}
	}
	
	int margin = 10;
	int barLength = 380;
	if (setBarLength && initBarLength > 10)
	{
		barLength = initBarLength;
	}
	double scale = double(barLength) / (maxVal - minVal);
	cv::Mat image = cv::Mat::zeros(barLength + 2 * margin, arrLength + 2 * margin, CV_8UC3);
	if (!showZero)
	{
		if (!showMean)
		{
			for (int i = 0; i < arrLength; i++)
			{
				cv::Point begin = cv::Point(i + margin, barLength + 2 * margin);
				cv::Point end = cv::Point(i + margin, barLength + margin - scale * (arr[i] - minVal));
				cv::line(image, begin, end, cv::Scalar::all(255));
			}
		}
		else
		{
			for (int i = 0; i < arrLength; i++)
			{
				cv::Point begin = cv::Point(i + margin, barLength + 2 * margin);
				cv::Point end = cv::Point(i + margin, barLength + margin - scale * (arr[i] - minVal));
				if (arr[i] < avg)
					cv::line(image, begin, end, cv::Scalar(255, 255, 0));
				else
					cv::line(image, begin, end, cv::Scalar(0, 255, 255));
			}
			int meanPosition = barLength + margin - scale * (avg - minVal);
			cv::line(image, cv::Point(margin, meanPosition), cv::Point(arrLength + margin, meanPosition), cv::Scalar(0, 0, 255));
		}
	}
	else
	{
		int zeroPosition = barLength + margin - scale * (0 - minVal);
		if (!showMean)
		{
			for (int i = 0; i < arrLength; i++)
			{
				cv::Point begin = cv::Point(i + margin, zeroPosition);
				cv::Point end = cv::Point(i + margin, barLength + margin - scale * (arr[i] - minVal));
				cv::line(image, begin, end, cv::Scalar::all(255));
			}
			cv::line(image, cv::Point(margin, zeroPosition), cv::Point(arrLength + margin, zeroPosition), cv::Scalar(255, 255, 255));
		}
		else
		{
			for (int i = 0; i < arrLength; i++)
			{
				cv::Point begin = cv::Point(i + margin, zeroPosition);
				cv::Point end = cv::Point(i + margin, barLength + margin - scale * (arr[i] - minVal));
				if (arr[i] < avg)
					cv::line(image, begin, end, cv::Scalar(255, 255, 0));
				else
					cv::line(image, begin, end, cv::Scalar(0, 255, 255));
			}
			int meanPosition = barLength + margin - scale * (avg - minVal);
			cv::line(image, cv::Point(margin, zeroPosition), cv::Point(arrLength + margin, zeroPosition), cv::Scalar(255, 255, 255));
			cv::line(image, cv::Point(margin, meanPosition), cv::Point(arrLength + margin, meanPosition), cv::Scalar(0, 0, 255));
		}
	}
	cv::imshow(name, image);
}

template<typename Type>
void showArrayByHoriBar(const std::string& name, const std::vector<Type>& arr, bool showMean, bool showZero, 
	bool setRange, double initMin, double initMax, bool setBarLength, double initBarLength)
{
	double minVal = arr[0];
	double maxVal = arr[0];
	double avg = 0;
	int arrLength = arr.size();
	for (int i = 0; i < arrLength; i++)
	{
		minVal = arr[i] < minVal ? arr[i] : minVal;
		maxVal = arr[i] > maxVal ? arr[i] : maxVal;
		avg += arr[i];
	}
	avg /= arrLength;	
	if (setRange && initMin < initMax)
	{
		minVal = initMin < minVal ? initMin : minVal;
		maxVal = initMax > maxVal ? initMax : maxVal;
	}
	if (showZero)
	{
		maxVal = maxVal < 0 ? 0 : maxVal;
		minVal = minVal > 0 ? 0 : minVal;
	}
	if (maxVal == minVal)
	{
		if (maxVal > 0)
		{
			minVal = 0;
			avg = maxVal;
		}
		else if (maxVal < 0)
		{
			maxVal = 0;
			avg = minVal;
		}
		else
		{
			maxVal = 1;
			minVal = 0;
			avg = 0;
		}
	}
	
	int margin = 10;
	int barLength = 380;
	if (setBarLength && initBarLength > 10)
	{
		barLength = initBarLength;
	}
	double scale = double(barLength) / (maxVal - minVal);
	cv::Mat image = cv::Mat::zeros(arrLength + 2 * margin, barLength + 2 * margin, CV_8UC3);
	if (!showZero)
	{
		if (!showMean)
		{
			for (int i = 0; i < arrLength; i++)
			{
				cv::Point begin = cv::Point(0, i + margin);
				cv::Point end = cv::Point(scale * (arr[i] - minVal) + margin, i + margin);
				cv::line(image, begin, end, cv::Scalar::all(255));
			}
		}
		else
		{
			for (int i = 0; i < arrLength; i++)
			{
				cv::Point begin = cv::Point(0, i + margin);
				cv::Point end = cv::Point(scale * (arr[i] - minVal), i + margin);
				if (arr[i] < avg)
					cv::line(image, begin, end, cv::Scalar(255, 255, 0));
				else
					cv::line(image, begin, end, cv::Scalar(0, 255, 255));
			}
			int meanPosition = scale * (avg - minVal) + margin;
			cv::line(image, cv::Point(meanPosition, margin), cv::Point(meanPosition, arrLength + margin), cv::Scalar(0, 0, 255));
		}
	}
	else
	{
		int zeroPosition = scale * (0 - minVal) + margin;
		if (!showMean)
		{
			for (int i = 0; i < arrLength; i++)
			{
				cv::Point begin = cv::Point(zeroPosition, i + margin);
				cv::Point end = cv::Point(scale * (arr[i] - minVal) + margin, i + margin);
				cv::line(image, begin, end, cv::Scalar::all(255));
			}
			cv::line(image, cv::Point(zeroPosition, margin), cv::Point(zeroPosition, arrLength + margin), cv::Scalar(255, 255, 255));
		}
		else
		{
			for (int i = 0; i < arrLength; i++)
			{
				cv::Point begin = cv::Point(zeroPosition, i + margin);
				cv::Point end = cv::Point(scale * (arr[i] - minVal) + margin, i + margin);
				if (arr[i] < avg)
					cv::line(image, begin, end, cv::Scalar(255, 255, 0));
				else
					cv::line(image, begin, end, cv::Scalar(0, 255, 255));
			}
			int meanPosition = scale * (avg - minVal) + margin;
			cv::line(image, cv::Point(zeroPosition, margin), cv::Point(zeroPosition, arrLength + margin), cv::Scalar(255, 255, 255));
			cv::line(image, cv::Point(meanPosition, margin), cv::Point(meanPosition, arrLength + margin), cv::Scalar(0, 0, 255));
		}
	}
	cv::imshow(name, image);
}

}
