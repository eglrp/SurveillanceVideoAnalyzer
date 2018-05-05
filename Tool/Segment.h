#pragma once
#include <vector>

namespace ztool
{

// 分段结构体，记录某个分段的值，这个值的起始下标(包含), 终止下标(包含)和长度
template <typename Type>
struct Segment
{
    int begin;
    int end;
    int length;
    Type data;
};

template <typename Type>
void findSegments(const Type* data, int dataLength, std::vector<Segment<Type> >& segments)
{
    segments.clear();
    if (dataLength <= 0) return;
    Segment<Type> currSegment;
    int currBegin;
    currBegin = 0;
    for (int i = 0; i < dataLength - 1; i++)
    {
        if (data[i] != data[i + 1])
        {
            currSegment.begin = currBegin;
            currSegment.end = i;
            currSegment.length = i - currBegin + 1;
            currSegment.data = data[i];
            segments.push_back(currSegment);
            currBegin = i + 1;
        }
    }
    currSegment.begin = currBegin;
    currSegment.end = dataLength - 1;
    currSegment.length = dataLength - currBegin;
    currSegment.data = data[currBegin];
    segments.push_back(currSegment);
}

template <typename Type>
void findSegments(const std::vector<Type>& data, std::vector<Segment<Type> >& segments)
{
    segments.clear();
    if (data.empty()) return;
    Segment<Type> currSegment;
    int dataLength = data.size();
    int currBegin;
    currBegin = 0;
    for (int i = 0; i < dataLength - 1; i++)
    {
        if (data[i] != data[i + 1])
        {
            currSegment.begin = currBegin;
            currSegment.end = i;
            currSegment.length = i - currBegin + 1;
            currSegment.data = data[i];
            segments.push_back(currSegment);
            currBegin = i + 1;
        }
    }
    currSegment.begin = currBegin;
    currSegment.end = dataLength - 1;
    currSegment.length = dataLength - currBegin;
    currSegment.data = data[currBegin];
    segments.push_back(currSegment);
}

template <typename DataType, typename ElemType, typename GetElemFunc>
void findSegments(const DataType* data, int dataLength, std::vector<Segment<ElemType> >& segments, GetElemFunc func)
{
    segments.clear();
    if (dataLength <= 0) return;
    Segment<ElemType> currSegment;
    int currBegin;
    currBegin = 0;
    for (int i = 0; i < dataLength - 1; i++)
    {
        if (func(data[i]) != func(data[i + 1]))
        {
            currSegment.begin = currBegin;
            currSegment.end = i;
            currSegment.length = i - currBegin + 1;
            currSegment.data = func(data[i]);
            segments.push_back(currSegment);
            currBegin = i + 1;
        }
    }
    currSegment.begin = currBegin;
    currSegment.end = dataLength - 1;
    currSegment.length = dataLength - currBegin;
    currSegment.data = func(data[currBegin]);
    segments.push_back(currSegment);
}

template <typename DataType, typename ElemType, typename GetElemFunc>
void findSegments(const std::vector<DataType>& data, std::vector<Segment<ElemType> >& segments, GetElemFunc func)
{
    segments.clear();
    if (data.empty()) return;
    Segment<ElemType> currSegment;
    int dataLength = data.size();
    int currBegin;
    currBegin = 0;
    for (int i = 0; i < dataLength - 1; i++)
    {
        if (func(data[i]) != func(data[i + 1]))
        {
            currSegment.begin = currBegin;
            currSegment.end = i;
            currSegment.length = i - currBegin + 1;
            currSegment.data = func(data[i]);
            segments.push_back(currSegment);
            currBegin = i + 1;
        }
    }
    currSegment.begin = currBegin;
    currSegment.end = dataLength - 1;
    currSegment.length = dataLength - currBegin;
    currSegment.data = func(data[currBegin]);
    segments.push_back(currSegment);
}

}