#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <exception>
#include "ProcVideo.h"

using namespace std;
using namespace zpv;

int main(int argc, char* argv[])
{
	const char* videoPath = /*"D:/SHARED/MiscellaneousVideo/E1-1_齐家园公寓（10）_20130816112945_20130816113000_23478187.mp4"*/
        "D:/SHARED/TaicangVideo/1/70.flv";
    double expectSegmentLengthInSecond = 300;
    double videoLengthInSecond;	
	vector<double> segmentLengthInSecond;
	vector<pair<int, int> > splitBegAndEnd;
    try
    {
        findSplitPositions(string(videoPath), expectSegmentLengthInSecond, 
            videoLengthInSecond, segmentLengthInSecond, splitBegAndEnd);
    }
    catch (const exception& e)
    {
        cout << e.what() << "\n";
        system("pause");
        return 0;
    }
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
