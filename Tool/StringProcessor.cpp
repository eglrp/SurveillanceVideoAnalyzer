#include <iostream>
#include <sstream>

#include "StringProcessor.h" 
#include "Segment.h"

using namespace std;

void ztool::cvtPathToFileName(const string& path, string& fileName)
{
	fileName.clear();
	fileName.reserve(path.size());
	/*bool isLastReplace = true;
	for (int i = 0; i < path.size(); i++)
	{
		if (path[i] == '\\' || 
			path[i] == '/' ||
			path[i] == ':' ||
			path[i] == '.')
		{
			if (!isLastReplace)
				fileName.push_back('_');
			isLastReplace = true;
		}
		else
		{
			fileName.push_back(path[i]);
			isLastReplace = false;
		}
	}*/

	vector<string> segs;
	ztool::splitString(path, segs, "\\/:.");
	for (int i = 0; i < segs.size() - 1; i++)
	{
		fileName += segs[i];
		fileName += '_';
	}
	fileName += segs[segs.size() - 1];
}

namespace
{
template<typename Type>
static inline Type stringToVal(const string& str)
{
    stringstream strm;
    Type val;
    strm << str.c_str();
    strm >> val;
    return val;
}
}

bool ztool::cvtStringToBool(const string& str)
{
    return stringToVal<bool>(str);
}

int ztool::cvtStringToInt(const string& str)
{
    return stringToVal<int>(str);
}

double ztool::cvtStringToDouble(const string& str)
{
    return stringToVal<double>(str);
}

void ztool::cvtStringToInts(const string& str, vector<int>& vals)
{
	vals.clear();
	vector<string> segs;
	splitString(str, segs, "-0123456789", false);
	if (segs.empty())
		return;

	vals.resize(segs.size());	
	for (int i = 0; i < segs.size(); i++)
	{
		stringstream tempStrm;
		tempStrm << segs[i].c_str();
		tempStrm >> vals[i];
	}
}

void ztool::cvtStringToDoubles(const string& str, vector<double>& vals)
{
	vals.clear();
	vector<string> segs;
	ztool::splitString(str, segs, ".-0123456789", false);
	if (segs.empty())
		return;

	vals.resize(segs.size());	
	for (int i = 0; i < segs.size(); i++)
	{
		stringstream tempStrm;
		tempStrm << segs[i].c_str();
		tempStrm >> vals[i];
	}
}

void ztool::splitString(const string& src, vector<string>& dst, const string& splitter, bool discastSplitter)
{
	dst.clear();
	if (src.size() < 1)
		return;

	int lenSrc = src.size();
	vector<bool> isReserve(lenSrc, false);
	int lenSplt = splitter.size();
	if (lenSplt == 0)
	{
		for (int i = 0; i < lenSrc; i++)
			isReserve[i] = isgraph(src[i]);
	}
	else
	{
		if (discastSplitter)
		{
			if (lenSplt == 1)
			{
				for (int i = 0; i < lenSrc; i++)
					isReserve[i] = (src[i] != splitter[0]);
			}
			else
			{
				for (int i = 0; i < lenSrc; i++)
				{
					int j = 0;
					for (; j < lenSplt; j++)
					{
						if (src[i] == splitter[j])
							break;
					}
					isReserve[i] = (j == lenSplt);
				}
			}
		}
		else
		{
			if (lenSplt == 1)
			{
				for (int i = 0; i < lenSrc; i++)
					isReserve[i] = (src[i] == splitter[0]);
			}
			else
			{
				for (int i = 0; i < lenSrc; i++)
				{
					int j = 0;
					for (; j < lenSplt; j++)
					{
						if (src[i] == splitter[j])
							break;
					}
					isReserve[i] = (j != lenSplt);
				}
			}
		}
	}

	vector<Segment<bool> > segs;
	findSegments(isReserve, segs);
	for (int i = 0; i < segs.size(); i++)
	{
		if (segs[i].data)
			dst.push_back(src.substr(segs[i].begin, segs[i].length));
	}
}