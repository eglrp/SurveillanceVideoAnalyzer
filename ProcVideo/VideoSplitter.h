#pragma once

#include <utility>
#include <vector>
#include "ExportControl.h"

namespace zpv
{

Z_LIB_EXPORT bool findSplitPositions(const std::string& videoPath, double splitUnitInSecond,
	double& videoLengthInSecond, std::vector<double>& segmentLengthInSecond,
	std::vector<std::pair<int, int> >& splitBegAndEnd);

}