#pragma once
#include <vector>
#include <string>

namespace ztool
{

void splitString(const std::string& src, std::vector<std::string>& dst, 
	const std::string& splitter = std::string(), bool discastSplitter = true);

void cvtPathToFileName(const std::string& path, std::string& fileName);

bool cvtStringToBool(const std::string& str);

int cvtStringToInt(const std::string& str);

double cvtStringToDouble(const std::string& str);

void cvtStringToInts(const std::string& str, std::vector<int>& vals);

void cvtStringToDoubles(const std::string& str, std::vector<double>& vals);

}