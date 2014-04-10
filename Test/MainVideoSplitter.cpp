#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include "VideoSplitter.h"

using namespace std;
using namespace zpv;

int main(int argc, char* argv[])
{
	const char* videoPath = "D:/SHARED/TaicangVideo/1/70.flv";
    double expectSegmentLengthInSecond = 600;
    double videoLengthInSecond;	
	vector<double> segmentLengthInSecond;
	vector<pair<int, int> > splitBegAndEnd;
    findSplitPositions(string(videoPath), expectSegmentLengthInSecond, 
        videoLengthInSecond, segmentLengthInSecond, splitBegAndEnd);
	cout << "In main :" << "\n";
	cout << "Split beginning and end in frame count:" << "\n";
	cout << fixed;
	cout << "video length in second: " << setprecision(4) << videoLengthInSecond << "\n";
	for (unsigned int i = 0; i < splitBegAndEnd.size(); i++)
	{
		cout << "segment:" << setw(4) << i << ", "
			 << "begin: " << setw(8) << splitBegAndEnd[i].first << ", "
		     << "end: " << setw(8) << splitBegAndEnd[i].second << ", "
		     << "length: " << setw(10) << setprecision(4) << segmentLengthInSecond[i] << "\n";
	}
    system("pause");
	return 0;
}
