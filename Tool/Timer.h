#pragma once

#include <opencv2/core/core.hpp>

namespace ztool
{

class Timer
{
public:
    Timer(void) 
    {
        freq = cv::getTickFrequency(); 
        startTime = cv::getTickCount();
    };
    void start(void) 
    {
        startTime = cv::getTickCount();
    };
    void end(void)
    {
        endTime = cv::getTickCount();
    }
    double elapse(void) 
    {
        return double(endTime - startTime) / freq;
    };
private:
    double freq;
    long long int startTime, endTime;
};

class RepeatTimer
{
public:
    RepeatTimer(void) 
    {
        freq = cv::getTickFrequency(); 
        clear();
    };
    void clear(void) 
    {
        startTime = 0; 
        accTime = 0; 
        count = 0;
    }
    void start(void) 
    {
        startTime = cv::getTickCount();
    };
    void end(void) 
    {
        accTime += (double(cv::getTickCount() - startTime) / freq); 
        count++;
    };
    double getAccTime(void) 
    {
        return accTime;
    };
    int getCount(void) 
    {
        return count;
    };
    double getAvgTime(void) 
    {
        return count == 0 ? 0 : (accTime / count);
    }
private:    
    double freq;
    long long int startTime;
    double accTime;
    int count;
};

}