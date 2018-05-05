#include <iomanip>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "MovingObjectDetector.h"
#include "CompileControl.h"
#include "CreateDirectory.h"

namespace zsfo
{

class OutputInfoParser::Impl
{
public:
    void init(const std::string& savePath, 
        const std::string& sceneImageName, const std::string& sliceImageName, const std::string& maskImageName,
        const std::string& objectInfoFileName, const std::string& objectHistoryFileName,
        bool isPicSmall = true, int waitKeyTime = 0);
    void show(const StampedImage& input, const ObjectDetails& output);
    void save(const ObjectDetails& output);
    void final(const std::string& label);

private:
    bool isFrameSmall;
    bool saveScene;
    bool saveSlice;
    bool saveMask;
    bool saveObjectInfo;
    bool saveObjectHistory;
    int waitTime;
    std::string resultPath;
    std::string sceneName, sliceName, maskName;
    std::string infoName;
    std::string historyName; 
    int infoCount;
    int historyCount;
    std::fstream objectInfoFile;
    std::fstream objectHistoryFile;
};

}

using namespace std;
using namespace cv;
using namespace ztool;

namespace zsfo
{

void OutputInfoParser::init(const string& savePath, 
    const string& sceneImageName, const string& sliceImageName, const string& maskImageName,
    const string& objectInfoFileName, const string& objectHistoryFileName,
    bool isPicSmall, int waitKeyTime)
{
    Impl* ptr = new Impl;
    ptrImpl = ptr;
    ptrImpl->init(savePath, sceneImageName, sliceImageName, maskImageName, 
        objectInfoFileName, objectHistoryFileName, isPicSmall, waitKeyTime); 
}

void OutputInfoParser::show(const StampedImage& input, const ObjectDetails& output)
{
    ptrImpl->show(input, output);
}

void OutputInfoParser::save(const ObjectDetails& output)
{
    ptrImpl->save(output);
}

void OutputInfoParser::final(const std::string& label)
{
    ptrImpl->final(label);
}

void OutputInfoParser::Impl::init(const string& savePath, 
    const string& sceneImageName, const string& sliceImageName, const string& maskImageName,
    const string& objectInfoFileName, const string& objectHistoryFileName,
    bool isPicSmall, int waitKeyTime)
{
    resultPath = savePath;
    createDirectory(resultPath);

    saveScene = sceneImageName.size();
    saveSlice = sliceImageName.size();
    saveMask = maskImageName.size();
    saveObjectInfo = objectInfoFileName.size();
    saveObjectHistory = objectHistoryFileName.size();

    sceneName = sceneImageName;
    sliceName = sliceImageName;
    maskName = maskImageName;
    infoName = objectInfoFileName;
    historyName = objectHistoryFileName;

    isFrameSmall = isPicSmall;
    waitTime = waitKeyTime;

    infoCount = 0;
    historyCount = 0;

    if (saveObjectInfo)
    {
        infoName = resultPath + "/" + objectInfoFileName;       
        objectInfoFile.open(infoName.c_str(), ios::out);
        objectInfoFile << "      ID        Time       Count       X       Y       W       H" << "\n";
        objectInfoFile.close();
    }

    if (saveObjectHistory)
    {
        historyName = resultPath + "/" + objectHistoryFileName;
        objectHistoryFile.open(historyName.c_str(), ios::out);
        objectHistoryFile.close();  
    }
}

}

namespace
{
template<typename Type>
static string getValString(Type val)
{
    stringstream strm;
    strm << val;
    return strm.str();
}
}

namespace zsfo
{

void OutputInfoParser::Impl::show(const StampedImage& input, const ObjectDetails& output)
{
    for (int i = 0; i < output.objects.size(); i++)
    {
#if CMPL_WRITE_CONSOLE
        printf("Blob ID: %d ", output.objects[i].ID);
#endif
        if (output.objects[i].isFinal)
        {
#if CMPL_WRITE_CONSOLE
            printf("speed: %.2f pixel per second, ", output.objects[i].speed);
            printf("velocity: %.2f kilometers per hour, ", output.objects[i].velocity);
#endif
#if CMPL_SHOW_IMAGE
            if (output.objects[i].hasSnapshotHistory)
            {
                int size = output.objects[i].snapshotHistory.size();
                for (int j = 0; j < size; j++)
                {
                    string indexStr = getValString(j);
                    const ObjectSnapshotRecord& refImage = output.objects[i].snapshotHistory[j];
                    if (refImage.scene.data)
                    {
                        Mat scene;
                        if (isFrameSmall)
                        {
                            scene = refImage.scene.clone();
                            rectangle(scene, refImage.rect, Scalar(0, 0, 255), 2);
                        }
                        else
                        {
                            resize(refImage.scene, scene, Size(refImage.scene.cols / 2, refImage.scene.rows / 2));
                            Rect rect = Rect(refImage.rect.x / 2, refImage.rect.y / 2,
                                refImage.rect.width / 2, refImage.rect.height / 2);
                            rectangle(scene, rect, Scalar(0, 0, 255), 2);           
                        }
                        imshow("Scene-" + indexStr, scene);
                    }
                    if (refImage.slice.data)
                    {
                        Mat slice;
                        if (isFrameSmall)
                            slice = refImage.slice.clone();
                        else
                            resize(refImage.slice, slice, Size(refImage.slice.cols / 2, refImage.slice.rows / 2)); 
                        imshow("Slice-" + indexStr, slice);
                    }
                    if (refImage.mask.data)
                    {
                        Mat mask;
                        if (isFrameSmall)
                            mask = refImage.mask.clone();
                        else
                            resize(refImage.mask, mask, Size(refImage.mask.cols / 2, refImage.mask.rows / 2));
                        imshow("Mask-" + indexStr, mask);
                    }
                }
                waitKey(waitTime);
                for (int j = 0; j < size; j++)
                {
                    string indexStr = getValString(j);
                    const ObjectSnapshotRecord& refImage = output.objects[i].snapshotHistory[j];
                    if (refImage.scene.data)
                        destroyWindow("Scene-" + indexStr);
                    if (refImage.slice.data)
                        destroyWindow("Slice-" + indexStr);
                    if (refImage.mask.data)
                        destroyWindow("Mask-" + indexStr);
                }
            }
#endif
        }
        else  // not final
        {
#if CMPL_WRITE_CONSOLE
            //printf("curr rect: x = %d, y = %d, w = %d, h = %d\n",
            //      output.objects[i].currRect.x, output.objects[i].currRect.y,
            //      output.objects[i].currRect.width, output.objects[i].currRect.height);
#endif
        }
    }

    for (int i = 0; i < output.staticObjects.size(); i++)
    {
#if CMPL_WRITE_CONSOLE
        printf("Static Blob ID: %d static time exceeds threshold\n", output.staticObjects[i].ID);
        printf("output image rect: x = %d, y = %d, w = %d, h = %d\n",
               output.staticObjects[i].rect.x, output.staticObjects[i].rect.y,
               output.staticObjects[i].rect.width, output.staticObjects[i].rect.height);
#endif
#if CMPL_SHOW_IMAGE
        Mat scene;
        namedWindow("Static Object");
        if (isFrameSmall)
        {
            scene = input.image.clone();
            rectangle(scene, output.staticObjects[i].rect, Scalar(0, 0, 255), 2);
        }
        else
        {           
            resize(input.image, scene, Size(input.image.cols / 2, input.image.rows / 2));
            Rect rect = Rect(output.staticObjects[i].rect.x / 2, output.staticObjects[i].rect.y / 2,
                output.staticObjects[i].rect.width / 2, output.staticObjects[i].rect.height / 2);
            rectangle(scene, rect, Scalar(0, 0, 255), 2);           
        }
        imshow("Static Object", scene);
        waitKey(waitTime);
        destroyWindow("Static Object");
#endif
    }
}

void OutputInfoParser::Impl::save(const ObjectDetails& output)
{
    /*if (!output.rects.empty())
    {
        printf("Foreground found:\n");
        for (int i = 0; i < output.rects.size(); i++)
        {
            printf("x = %d, y = %d, w = %d, h = %d.\n",
                    output.rects[i].x, output.rects[i].y,
                    output.rects[i].width, output.rects[i].height);
        }
    }*/
    if (saveObjectInfo)
    {
        objectInfoFile.open(infoName.c_str(), ios::ate | ios::out | ios::in);
        objectInfoFile << fixed;
    }
    if (saveObjectHistory)
        objectHistoryFile.open(historyName.c_str(), ios::ate | ios::out | ios::in);
    for (int i = 0; i < output.objects.size(); i++)
    {
        if (!output.objects[i].isFinal)
            continue;

        const ObjectInfo& refObject = output.objects[i];

        if (output.objects[i].hasSnapshotHistory)
        {
            infoCount++;
            int size = output.objects[i].snapshotHistory.size();
            for (int j = 0; j < size; j++)
            {
                stringstream sceneNameStr, sliceNameStr, maskNameStr;
                const ObjectSnapshotRecord refImage = refObject.snapshotHistory[j];
                if (saveScene && refImage.scene.data)
                {
                    sceneNameStr << resultPath << "/" << sceneName << refObject.ID << "-" << j << ".jpg";
                    imwrite(sceneNameStr.str(), refImage.scene);
                }
                if (saveSlice && refImage.slice.data)
                {
                    sliceNameStr << resultPath << "/" << sliceName << refObject.ID << "-" << j << ".jpg";
                    imwrite(sliceNameStr.str(), refImage.slice);
                }
                if (saveMask && refImage.mask.data)
                {
                    maskNameStr << resultPath << "/" << maskName << refObject.ID << "-" << j << ".jpg";
                    imwrite(maskNameStr.str(), refImage.mask);  
                }
                if (saveObjectInfo)
                {
                    objectInfoFile << setw(8) << refObject.ID
                        << setw(12) << refImage.time
                        << setw(12) << refImage.number
                        << setw(8) << refImage.rect.x << setw(8) << refImage.rect.y
                        << setw(8) << refImage.rect.width << setw(8) << refImage.rect.height;
                    objectInfoFile << "\n";
                }
            }
        }

        if (saveObjectHistory)
        {
            if (output.objects[i].hasHistory)
            {
                historyCount++; 
                const vector<ObjectRecord>& refHistory = refObject.history;
                objectHistoryFile << "Vehicle Count: " << historyCount << "\n";
                objectHistoryFile << "ID:            " << refObject.ID << "\n";
                objectHistoryFile << "Size:          " << refHistory.size() << "\n";
                objectHistoryFile << "Frame Count Time Stamp       x       y       w       h" << "\n";
                for (int j = 0; j < output.objects[i].history.size(); j++)
                {
                    objectHistoryFile << setw(11) << refHistory[j].number
                        << setw(11) << refHistory[j].time
                        << setw(8) << refHistory[j].normRect.x
                        << setw(8) << refHistory[j].normRect.y
                        << setw(8) << refHistory[j].normRect.width
                        << setw(8) << refHistory[j].normRect.height << "\n";
                }
                objectHistoryFile << "\n";
            }   
        }
    }
    if (saveObjectInfo)
        objectInfoFile.close();
    if (saveObjectHistory)
        objectHistoryFile.close();
} 

void OutputInfoParser::Impl::final(const string& label)
{
    if (saveObjectInfo)
    {
        objectInfoFile.open(infoName.c_str(), ios::ate | ios::out | ios::in);
        objectInfoFile << label << "\n";
        objectInfoFile.close();
    }

    if (saveObjectHistory)
    {
        objectHistoryFile.open(historyName.c_str(), ios::ate | ios::out | ios::in);
        objectHistoryFile << label << "\n";
        objectHistoryFile.close();
    }
}

}