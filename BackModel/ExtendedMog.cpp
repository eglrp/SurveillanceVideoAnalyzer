#include "ExtendedMog.h"
#include "Exception.h"

using namespace cv;
using namespace std;

#define CMPL_MOG_USE_RELAXED_PARAM 1
#define CMPL_MOG_USE_APPROX_THRESHOLD 1

#if CMPL_MOG_USE_RELAXED_PARAM
static const int numGauss = 4;
static const int maxCount = 1000;
static const float minLearnRate = 1.0F / maxCount;
static const float thresForeBack = 0.7;
static const float thresSqrMahaDist = 2.5 * 2.5;
static const float initGaussStdDev = 30;
static const float initGaussVar = 30 * 30;
static const float minGaussStdDev = 15;
static const float minGaussVar = 15 * 15;
static const float initWeight = 0.05;
#else
static const int numGauss = 4;
static const int maxCount = 1000;
static const float minLearnRate = 1.0F / maxCount;
static const float thresForeBack = 0.7;
static const float thresSqrMahaDist = 2.5 * 2.5;
static const float initGaussStdDev = 15;
static const float initGaussVar = 15 * 15;
static const float minGaussStdDev = 8;
static const float minGaussVar = 8 * 8;
static const float initWeight = 0.05;
#endif

namespace
{
struct ModelC1
{
    float mean, var, weight;
};

struct ModelC3
{
    float meanB, meanG, meanR, varB, varG, varR, weight;
};
}

void zsfo::Mog::init(const Mat& image)
{
    if (image.rows <= 0 || image.cols <= 0 || image.type() != CV_8UC1 && image.type() != CV_8UC3)
        THROW_EXCEPT("input image format not supported");

    width = image.cols, height = image.rows;
    type = image.type();
    count = 1;
    if (type == CV_8UC1)
        model.create(height, width * sizeof(ModelC1) * numGauss, CV_8UC1);
    else if (type == CV_8UC3)
        model.create(height, width * sizeof(ModelC3) * numGauss, CV_8UC1);
    model.setTo(0);
    mask.create(height, width, CV_8UC1);
    mask.setTo(0);
    float weight = 1.0F / numGauss;
    if (type == CV_8UC1)
    {
        for (int i = 0; i < height; i++)
        {
            const unsigned char* ptrImage = image.ptr<unsigned char>(i);
            ModelC1* ptrModel = (ModelC1*)model.ptr<unsigned char>(i);
            for (int j = 0; j < width; j++)
            {
                for (int k = 0; k < numGauss; k++)
                {
                    ptrModel->mean = *ptrImage;
                    ptrModel->var = initGaussVar;
                    ptrModel->weight = weight;
                    ptrModel++;
                }
                ptrImage++;
            }
        }
    }
    else if (type == CV_8UC3)
    {
        for (int i = 0; i < height; i++)
        {
            const unsigned char* ptrImage = image.ptr<unsigned char>(i);
            ModelC3* ptrModel = (ModelC3*)model.ptr<unsigned char>(i);
            for (int j = 0; j < width; j++)
            {
                for (int k = 0; k < numGauss; k++)
                {
                    ptrModel->meanB = ptrImage[0];
                    ptrModel->meanG = ptrImage[1];
                    ptrModel->meanR = ptrImage[2];
                    ptrModel->varB = initGaussVar;
                    ptrModel->varG = initGaussVar;
                    ptrModel->varR = initGaussVar;
                    ptrModel->weight = weight;
                    ptrModel++;
                }
                ptrImage += 3;
            }
        }
    }
}

void zsfo::Mog::update(const Mat& image, Mat& fore, Mat& back, const vector<Rect>& noUpdate)
{
    if (image.cols != width && image.rows != height && image.type() != type)
        THROW_EXCEPT("input image size or format not valid");
    
    fore.create(height, width, CV_8UC1);
    back.create(height, width, type);
    float learnRate = minLearnRate;
    if (count < maxCount)
    {
        count++;
        learnRate = 1.0F / count;
    }
    memset(mask.data, 255, width * height);
    if (!noUpdate.empty())
    {
        int numRect = noUpdate.size();
        Rect base(0, 0, width, height);
        for (int i = 0; i < numRect; i++)
        {
            Rect currRect = base & noUpdate[i];
            mask(currRect).setTo(0);
        }
    }
    if (type == CV_8UC1)
    {
        for (int i = 0; i < height; i++)
        {
            const unsigned char* ptrImage = image.ptr<unsigned char>(i);
            const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
            unsigned char* ptrFore = fore.ptr<unsigned char>(i);
            unsigned char* ptrBack = back.ptr<unsigned char>(i);
            ModelC1* ptrModel = (ModelC1*)model.ptr<unsigned char>(i);
            for (int j = 0; j < width; j++)
            {
                ModelC1* ptrCurrModel = ptrModel + j * numGauss;
                float val = ptrImage[j];
                float currDiff;
                float currDist;
                int index = -1;
                for (int k = 0; k < numGauss; k++)
                {
                    currDiff = val - ptrCurrModel[k].mean;
                    currDist = currDiff * currDiff / ptrCurrModel[k].var;
                    if (currDist < thresSqrMahaDist)
                    {
                        index = k;
                        break;
                    }
                }
                if (!ptrMask[j])
                {
                    if (index != -1)
                    {
                        float weightSum = 0;
                        int indexBack = 0;
                        for (int k = 0; k < numGauss; k++)
                        {
                            weightSum += ptrCurrModel[k].weight;
                            if (weightSum > thresForeBack)
                            {
                                indexBack = k;
                                break;
                            }
                        }
                        ptrFore[j] = (index > indexBack) ? 255 : 0;
                    }
                    else
                        ptrFore[j] = 255;
                    ptrBack[j] = ptrCurrModel[0].mean;
                    continue;
                }
                if (index != -1)
                {
                    for (int k = 0; k < numGauss; k++)
                    {
                        if (index != k)
                            ptrCurrModel[k].weight -= learnRate * ptrCurrModel[k].weight;
                        else
                        {
                            ptrCurrModel[k].mean += learnRate * currDiff;
                            ptrCurrModel[k].var += learnRate * (currDiff * currDiff - ptrCurrModel[k].var);
                            if (ptrCurrModel[k].var < minGaussVar) 
                                ptrCurrModel[k].var = minGaussVar; 
                            ptrCurrModel[k].weight += learnRate * (1.0F - ptrCurrModel[k].weight);
                        }
                    }
                    for (int k = index -1; k >= 0; k--)
                    {
                        if (ptrCurrModel[k].weight < ptrCurrModel[k + 1].weight)
                        {
                            index = k;
                            ModelC1 temp = ptrCurrModel[k];
                            ptrCurrModel[k] = ptrCurrModel[k + 1];
                            ptrCurrModel[k + 1] = temp; 
                        }
                    }
                    float weightSum = 0;
                    for (int k = 0; k < numGauss; k++)
                        weightSum += ptrCurrModel[k].weight;
                    if (weightSum > 1.0001F || weightSum < 0.9999F)
                    {
                        weightSum = 1.0F / weightSum;
                        for (int k = 0; k < numGauss; k++)
                            ptrCurrModel[k].weight *= weightSum;
                    }
                    weightSum = 0;
                    int indexBack = 0;
                    for (int k = 0; k < numGauss; k++)
                    {
                        weightSum += ptrCurrModel[k].weight;
                        if (weightSum > thresForeBack)
                        {
                            indexBack = k;
                            break;
                        }
                    }
                    ptrFore[j] = (index > indexBack) ? 255 : 0;
                }
                else
                {
                    ptrFore[j] = 255;
                    ptrCurrModel[numGauss - 1].mean = val;
                    ptrCurrModel[numGauss - 1].var = initGaussVar;
                    ptrCurrModel[numGauss - 1].weight = initWeight;
                    float weightSum = 0;
                    for (int k = 0; k < numGauss; k++)
                        weightSum += ptrCurrModel[k].weight;
                    weightSum = 1.0F / weightSum;
                    for (int k = 0; k < numGauss; k++)
                        ptrCurrModel[k].weight *= weightSum;
                    for (int k = numGauss - 2; k >= 0; k--)
                    {
                        if (ptrCurrModel[k].weight < ptrCurrModel[k + 1].weight)
                        {
                            index = k;
                            ModelC1 temp = ptrCurrModel[k];
                            ptrCurrModel[k] = ptrCurrModel[k + 1];
                            ptrCurrModel[k + 1] = temp; 
                        }
                    }
                }
                ptrBack[j] = ptrCurrModel[0].mean;
            }
        }
    }
    else if (type == CV_8UC3)
    {
        for (int i = 0; i < height; i++)
        {
            const unsigned char* ptrImage = image.ptr<unsigned char>(i);
            const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
            unsigned char* ptrFore = fore.ptr<unsigned char>(i);
            unsigned char* ptrBack = back.ptr<unsigned char>(i);
            ModelC3* ptrModel = (ModelC3*)model.ptr<unsigned char>(i);
            for (int j = 0; j < width; j++)
            {
                ModelC3* ptrCurrModel = ptrModel + j * numGauss;
                float valB = ptrImage[j * 3], valG = ptrImage[j * 3 + 1], valR = ptrImage[j * 3 + 2];
                float currDiffB;
                float currDiffG;
                float currDiffR;
                float currDist;
                int index = -1;
                for (int k = 0; k < numGauss; k++)
                {
                    currDiffB = valB - ptrCurrModel[k].meanB;
                    currDiffG = valG - ptrCurrModel[k].meanG;
                    currDiffR = valR - ptrCurrModel[k].meanR;
#if CMPL_MOG_USE_APPROX_THRESHOLD
                    currDist = (currDiffB * currDiffB + currDiffG * currDiffG + currDiffR * currDiffR) / 
                               (ptrCurrModel[k].varB + ptrCurrModel[k].varG + ptrCurrModel[k].varR);
#else
                    currDist = currDiffB * currDiffB / ptrCurrModel[k].varB + 
                               currDiffG * currDiffG / ptrCurrModel[k].varG + 
                               currDiffR * currDiffR / ptrCurrModel[k].varR;
#endif
                    if (currDist < thresSqrMahaDist)
                    {
                        index = k;
                        break;
                    }
                }
                if (!ptrMask[j])
                {
                    if (index != -1)
                    {
                        float weightSum = 0;
                        int indexBack = 0;
                        for (int k = 0; k < numGauss; k++)
                        {
                            weightSum += ptrCurrModel[k].weight;
                            if (weightSum > thresForeBack)
                            {
                                indexBack = k;
                                break;
                            }
                        }
                        ptrFore[j] = (index > indexBack) ? 255 : 0;
                    }
                    else
                        ptrFore[j] = 255;
                    ptrBack[j * 3] = ptrCurrModel[0].meanB;
                    ptrBack[j * 3 + 1] = ptrCurrModel[0].meanG;
                    ptrBack[j * 3 + 2] = ptrCurrModel[0].meanR;
                    continue;
                }
                if (index != -1)
                {
                    for (int k = 0; k < numGauss; k++)
                    {
                        if (index != k)
                            ptrCurrModel[k].weight -= learnRate * ptrCurrModel[k].weight;
                        else
                        {
                            ptrCurrModel[k].meanB += learnRate * currDiffB;
                            ptrCurrModel[k].meanG += learnRate * currDiffG;
                            ptrCurrModel[k].meanR += learnRate * currDiffR;
                            ptrCurrModel[k].varB += learnRate * (currDiffB * currDiffB - ptrCurrModel[k].varB);
                            ptrCurrModel[k].varG += learnRate * (currDiffG * currDiffG - ptrCurrModel[k].varG);
                            ptrCurrModel[k].varR += learnRate * (currDiffR * currDiffR - ptrCurrModel[k].varR);
                            if (ptrCurrModel[k].varB < minGaussVar) 
                                ptrCurrModel[k].varB = minGaussVar; 
                            if (ptrCurrModel[k].varG < minGaussVar) 
                                ptrCurrModel[k].varG = minGaussVar; 
                            if (ptrCurrModel[k].varR < minGaussVar) 
                                ptrCurrModel[k].varR = minGaussVar; 
                            ptrCurrModel[k].weight += learnRate * (1.0F - ptrCurrModel[k].weight);
                        }
                    }
                    for (int k = index -1; k >= 0; k--)
                    {
                        if (ptrCurrModel[k].weight < ptrCurrModel[k + 1].weight)
                        {
                            index = k;
                            ModelC3 temp = ptrCurrModel[k];
                            ptrCurrModel[k] = ptrCurrModel[k + 1];
                            ptrCurrModel[k + 1] = temp; 
                        }
                    }
                    float weightSum = 0;
                    for (int k = 0; k < numGauss; k++)
                        weightSum += ptrCurrModel[k].weight;
                    if (weightSum > 1.0001F || weightSum < 0.9999F)
                    {
                        weightSum = 1.0F / weightSum;
                        for (int k = 0; k < numGauss; k++)
                            ptrCurrModel[k].weight *= weightSum;
                    }
                    weightSum = 0;
                    int indexBack = 0;
                    for (int k = 0; k < numGauss; k++)
                    {
                        weightSum += ptrCurrModel[k].weight;
                        if (weightSum > thresForeBack)
                        {
                            indexBack = k;
                            break;
                        }
                    }
                    ptrFore[j] = (index > indexBack) ? 255 : 0;
                }
                else
                {
                    ptrFore[j] = 255;
                    ptrCurrModel[numGauss - 1].meanB = valB;
                    ptrCurrModel[numGauss - 1].meanG = valG;
                    ptrCurrModel[numGauss - 1].meanR = valR;
                    ptrCurrModel[numGauss - 1].varB = initGaussVar;
                    ptrCurrModel[numGauss - 1].varG = initGaussVar;
                    ptrCurrModel[numGauss - 1].varR = initGaussVar;
                    ptrCurrModel[numGauss - 1].weight = initWeight;
                    float weightSum = 0;
                    for (int k = 0; k < numGauss; k++)
                        weightSum += ptrCurrModel[k].weight;
                    weightSum = 1.0F / weightSum;
                    for (int k = 0; k < numGauss; k++)
                        ptrCurrModel[k].weight *= weightSum;
                    for (int k = numGauss - 2; k >= 0; k--)
                    {
                        if (ptrCurrModel[k].weight < ptrCurrModel[k + 1].weight)
                        {
                            index = k;
                            ModelC3 temp = ptrCurrModel[k];
                            ptrCurrModel[k] = ptrCurrModel[k + 1];
                            ptrCurrModel[k + 1] = temp; 
                        }
                    }
                }
                ptrBack[j * 3] = ptrCurrModel[0].meanB;
                ptrBack[j * 3 + 1] = ptrCurrModel[0].meanG;
                ptrBack[j * 3 + 2] = ptrCurrModel[0].meanR;
            }
        }
    }
}

void zsfo::Mog::update(const Mat& image, Mat& fore, const vector<Rect>& noUpdate)
{
    if (image.cols != width && image.rows != height && image.type() != type)
        THROW_EXCEPT("input image size or format not valid");

    fore.create(height, width, CV_8UC1);
    float learnRate = minLearnRate;
    if (count < maxCount)
    {
        count++;
        learnRate = 1.0F / count;
    }
    memset(mask.data, 255, width * height);
    if (!noUpdate.empty())
    {
        int numRect = noUpdate.size();
        Rect base(0, 0, width, height);
        for (int i = 0; i < numRect; i++)
        {
            Rect currRect = base & noUpdate[i];
            mask(currRect).setTo(0);
        }
    }
    if (type == CV_8UC1)
    {
        for (int i = 0; i < height; i++)
        {
            const unsigned char* ptrImage = image.ptr<unsigned char>(i);
            const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
            unsigned char* ptrFore = fore.ptr<unsigned char>(i);
            ModelC1* ptrModel = (ModelC1*)model.ptr<unsigned char>(i);
            for (int j = 0; j < width; j++)
            {
                ModelC1* ptrCurrModel = ptrModel + j * numGauss;
                float val = ptrImage[j];
                float currDiff;
                float currDist;
                int index = -1;
                for (int k = 0; k < numGauss; k++)
                {
                    currDiff = val - ptrCurrModel[k].mean;
                    currDist = currDiff * currDiff / ptrCurrModel[k].var;
                    if (currDist < thresSqrMahaDist)
                    {
                        index = k;
                        break;
                    }
                }
                if (!ptrMask[j])
                {
                    if (index != -1)
                    {
                        float weightSum = 0;
                        int indexBack = 0;
                        for (int k = 0; k < numGauss; k++)
                        {
                            weightSum += ptrCurrModel[k].weight;
                            if (weightSum > thresForeBack)
                            {
                                indexBack = k;
                                break;
                            }
                        }
                        ptrFore[j] = (index > indexBack) ? 255 : 0;
                    }
                    else
                        ptrFore[j] = 255;
                    continue;
                }
                if (index != -1)
                {
                    for (int k = 0; k < numGauss; k++)
                    {
                        if (index != k)
                            ptrCurrModel[k].weight -= learnRate * ptrCurrModel[k].weight;
                        else
                        {
                            ptrCurrModel[k].mean += learnRate * currDiff;
                            ptrCurrModel[k].var += learnRate * (currDiff * currDiff - ptrCurrModel[k].var);
                            if (ptrCurrModel[k].var < minGaussVar) 
                                ptrCurrModel[k].var = minGaussVar; 
                            ptrCurrModel[k].weight += learnRate * (1.0F - ptrCurrModel[k].weight);
                        }
                    }
                    for (int k = index -1; k >= 0; k--)
                    {
                        if (ptrCurrModel[k].weight < ptrCurrModel[k + 1].weight)
                        {
                            index = k;
                            ModelC1 temp = ptrCurrModel[k];
                            ptrCurrModel[k] = ptrCurrModel[k + 1];
                            ptrCurrModel[k + 1] = temp; 
                        }
                    }
                    float weightSum = 0;
                    for (int k = 0; k < numGauss; k++)
                        weightSum += ptrCurrModel[k].weight;
                    if (weightSum > 1.0001F || weightSum < 0.9999F)
                    {
                        weightSum = 1.0F / weightSum;
                        for (int k = 0; k < numGauss; k++)
                            ptrCurrModel[k].weight *= weightSum;
                    }
                    weightSum = 0;
                    int indexBack = 0;
                    for (int k = 0; k < numGauss; k++)
                    {
                        weightSum += ptrCurrModel[k].weight;
                        if (weightSum > thresForeBack)
                        {
                            indexBack = k;
                            break;
                        }
                    }
                    ptrFore[j] = (index > indexBack) ? 255 : 0;
                }
                else
                {
                    ptrFore[j] = 255;
                    ptrCurrModel[numGauss - 1].mean = val;
                    ptrCurrModel[numGauss - 1].var = initGaussVar;
                    ptrCurrModel[numGauss - 1].weight = initWeight;
                    float weightSum = 0;
                    for (int k = 0; k < numGauss; k++)
                        weightSum += ptrCurrModel[k].weight;
                    weightSum = 1.0F / weightSum;
                    for (int k = 0; k < numGauss; k++)
                        ptrCurrModel[k].weight *= weightSum;
                    for (int k = numGauss - 2; k >= 0; k--)
                    {
                        if (ptrCurrModel[k].weight < ptrCurrModel[k + 1].weight)
                        {
                            index = k;
                            ModelC1 temp = ptrCurrModel[k];
                            ptrCurrModel[k] = ptrCurrModel[k + 1];
                            ptrCurrModel[k + 1] = temp; 
                        }
                    }
                }
            }
        }
    }
    else if (type == CV_8UC3)
    {
        for (int i = 0; i < height; i++)
        {
            const unsigned char* ptrImage = image.ptr<unsigned char>(i);
            const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
            unsigned char* ptrFore = fore.ptr<unsigned char>(i);
            ModelC3* ptrModel = (ModelC3*)model.ptr<unsigned char>(i);
            for (int j = 0; j < width; j++)
            {
                ModelC3* ptrCurrModel = ptrModel + j * numGauss;
                float valB = ptrImage[j * 3], valG = ptrImage[j * 3 + 1], valR = ptrImage[j * 3 + 2];
                float currDiffB;
                float currDiffG;
                float currDiffR;
                float currDist;
                int index = -1;
                for (int k = 0; k < numGauss; k++)
                {
                    currDiffB = valB - ptrCurrModel[k].meanB;
                    currDiffG = valG - ptrCurrModel[k].meanG;
                    currDiffR = valR - ptrCurrModel[k].meanR;
#if CMPL_MOG_USE_APPROX_THRESHOLD
                    currDist = (currDiffB * currDiffB + currDiffG * currDiffG + currDiffR * currDiffR) / 
                               (ptrCurrModel[k].varB + ptrCurrModel[k].varG + ptrCurrModel[k].varR);
#else
                    currDist = currDiffB * currDiffB / ptrCurrModel[k].varB + 
                               currDiffG * currDiffG / ptrCurrModel[k].varG + 
                               currDiffR * currDiffR / ptrCurrModel[k].varR;
#endif
                    if (currDist < thresSqrMahaDist)
                    {
                        index = k;
                        break;
                    }
                }
                if (!ptrMask[j])
                {
                    if (index != -1)
                    {
                        float weightSum = 0;
                        int indexBack = 0;
                        for (int k = 0; k < numGauss; k++)
                        {
                            weightSum += ptrCurrModel[k].weight;
                            if (weightSum > thresForeBack)
                            {
                                indexBack = k;
                                break;
                            }
                        }
                        ptrFore[j] = (index > indexBack) ? 255 : 0;
                    }
                    else
                        ptrFore[j] = 255;
                    continue;
                }
                if (index != -1)
                {
                    for (int k = 0; k < numGauss; k++)
                    {
                        if (index != k)
                            ptrCurrModel[k].weight -= learnRate * ptrCurrModel[k].weight;
                        else
                        {
                            ptrCurrModel[k].meanB += learnRate * currDiffB;
                            ptrCurrModel[k].meanG += learnRate * currDiffG;
                            ptrCurrModel[k].meanR += learnRate * currDiffR;
                            ptrCurrModel[k].varB += learnRate * (currDiffB * currDiffB - ptrCurrModel[k].varB);
                            ptrCurrModel[k].varG += learnRate * (currDiffG * currDiffG - ptrCurrModel[k].varG);
                            ptrCurrModel[k].varR += learnRate * (currDiffR * currDiffR - ptrCurrModel[k].varR);
                            if (ptrCurrModel[k].varB < minGaussVar) 
                                ptrCurrModel[k].varB = minGaussVar; 
                            if (ptrCurrModel[k].varG < minGaussVar) 
                                ptrCurrModel[k].varG = minGaussVar; 
                            if (ptrCurrModel[k].varR < minGaussVar) 
                                ptrCurrModel[k].varR = minGaussVar; 
                            ptrCurrModel[k].weight += learnRate * (1.0F - ptrCurrModel[k].weight);
                        }
                    }
                    for (int k = index -1; k >= 0; k--)
                    {
                        if (ptrCurrModel[k].weight < ptrCurrModel[k + 1].weight)
                        {
                            index = k;
                            ModelC3 temp = ptrCurrModel[k];
                            ptrCurrModel[k] = ptrCurrModel[k + 1];
                            ptrCurrModel[k + 1] = temp; 
                        }
                    }
                    float weightSum = 0;
                    for (int k = 0; k < numGauss; k++)
                        weightSum += ptrCurrModel[k].weight;
                    if (weightSum > 1.0001F || weightSum < 0.9999F)
                    {
                        weightSum = 1.0F / weightSum;
                        for (int k = 0; k < numGauss; k++)
                            ptrCurrModel[k].weight *= weightSum;
                    }
                    weightSum = 0;
                    int indexBack = 0;
                    for (int k = 0; k < numGauss; k++)
                    {
                        weightSum += ptrCurrModel[k].weight;
                        if (weightSum > thresForeBack)
                        {
                            indexBack = k;
                            break;
                        }
                    }
                    ptrFore[j] = (index > indexBack) ? 255 : 0;
                }
                else
                {
                    ptrFore[j] = 255;
                    ptrCurrModel[numGauss - 1].meanB = valB;
                    ptrCurrModel[numGauss - 1].meanG = valG;
                    ptrCurrModel[numGauss - 1].meanR = valR;
                    ptrCurrModel[numGauss - 1].varB = initGaussVar;
                    ptrCurrModel[numGauss - 1].varG = initGaussVar;
                    ptrCurrModel[numGauss - 1].varR = initGaussVar;
                    ptrCurrModel[numGauss - 1].weight = initWeight;
                    float weightSum = 0;
                    for (int k = 0; k < numGauss; k++)
                        weightSum += ptrCurrModel[k].weight;
                    weightSum = 1.0F / weightSum;
                    for (int k = 0; k < numGauss; k++)
                        ptrCurrModel[k].weight *= weightSum;
                    for (int k = numGauss - 2; k >= 0; k--)
                    {
                        if (ptrCurrModel[k].weight < ptrCurrModel[k + 1].weight)
                        {
                            index = k;
                            ModelC3 temp = ptrCurrModel[k];
                            ptrCurrModel[k] = ptrCurrModel[k + 1];
                            ptrCurrModel[k + 1] = temp; 
                        }
                    }
                }
            }
        }
    }
}

void zsfo::Mog::update(const Mat& image, const vector<Rect>& noUpdate)
{
    if (image.cols != width && image.rows != height && image.type() != type)
        THROW_EXCEPT("input image size or format not valid");

    float learnRate = minLearnRate;
    if (count < maxCount)
    {
        count++;
        learnRate = 1.0F / count;
    }
    memset(mask.data, 255, width * height);
    if (!noUpdate.empty())
    {
        int numRect = noUpdate.size();
        Rect base(0, 0, width, height);
        for (int i = 0; i < numRect; i++)
        {
            Rect currRect = base & noUpdate[i];
            mask(currRect).setTo(0);
        }
    }
    if (type == CV_8UC1)
    {
        for (int i = 0; i < height; i++)
        {
            const unsigned char* ptrImage = image.ptr<unsigned char>(i);
            const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
            ModelC1* ptrModel = (ModelC1*)model.ptr<unsigned char>(i);
            for (int j = 0; j < width; j++)
            {
                ModelC1* ptrCurrModel = ptrModel + j * numGauss;
                float val = ptrImage[j];
                float currDiff;
                float currDist;
                int index = -1;
                for (int k = 0; k < numGauss; k++)
                {
                    currDiff = val - ptrCurrModel[k].mean;
                    currDist = currDiff * currDiff / ptrCurrModel[k].var;
                    if (currDist < thresSqrMahaDist)
                    {
                        index = k;
                        break;
                    }
                }
                if (!ptrMask[j])
                    continue;
                if (index != -1)
                {
                    for (int k = 0; k < numGauss; k++)
                    {
                        if (index != k)
                            ptrCurrModel[k].weight -= learnRate * ptrCurrModel[k].weight;
                        else
                        {
                            ptrCurrModel[k].mean += learnRate * currDiff;
                            ptrCurrModel[k].var += learnRate * (currDiff * currDiff - ptrCurrModel[k].var);
                            if (ptrCurrModel[k].var < minGaussVar) 
                                ptrCurrModel[k].var = minGaussVar; 
                            ptrCurrModel[k].weight += learnRate * (1.0F - ptrCurrModel[k].weight);
                        }
                    }
                    for (int k = index -1; k >= 0; k--)
                    {
                        if (ptrCurrModel[k].weight < ptrCurrModel[k + 1].weight)
                        {
                            index = k;
                            ModelC1 temp = ptrCurrModel[k];
                            ptrCurrModel[k] = ptrCurrModel[k + 1];
                            ptrCurrModel[k + 1] = temp; 
                        }
                    }
                    float weightSum = 0;
                    for (int k = 0; k < numGauss; k++)
                        weightSum += ptrCurrModel[k].weight;
                    if (weightSum > 1.0001F || weightSum < 0.9999F)
                    {
                        weightSum = 1.0F / weightSum;
                        for (int k = 0; k < numGauss; k++)
                            ptrCurrModel[k].weight *= weightSum;
                    }
                }
                else
                {
                    ptrCurrModel[numGauss - 1].mean = val;
                    ptrCurrModel[numGauss - 1].var = initGaussVar;
                    ptrCurrModel[numGauss - 1].weight = initWeight;
                    float weightSum = 0;
                    for (int k = 0; k < numGauss; k++)
                        weightSum += ptrCurrModel[k].weight;
                    weightSum = 1.0F / weightSum;
                    for (int k = 0; k < numGauss; k++)
                        ptrCurrModel[k].weight *= weightSum;
                    for (int k = numGauss - 2; k >= 0; k--)
                    {
                        if (ptrCurrModel[k].weight < ptrCurrModel[k + 1].weight)
                        {
                            index = k;
                            ModelC1 temp = ptrCurrModel[k];
                            ptrCurrModel[k] = ptrCurrModel[k + 1];
                            ptrCurrModel[k + 1] = temp; 
                        }
                    }
                }
            }
        }
    }
    else if (type == CV_8UC3)
    {
        for (int i = 0; i < height; i++)
        {
            const unsigned char* ptrImage = image.ptr<unsigned char>(i);
            const unsigned char* ptrMask = mask.ptr<unsigned char>(i);
            ModelC3* ptrModel = (ModelC3*)model.ptr<unsigned char>(i);
            for (int j = 0; j < width; j++)
            {
                ModelC3* ptrCurrModel = ptrModel + j * numGauss;
                float valB = ptrImage[j * 3], valG = ptrImage[j * 3 + 1], valR = ptrImage[j * 3 + 2];
                float currDiffB;
                float currDiffG;
                float currDiffR;
                float currDist;
                int index = -1;
                for (int k = 0; k < numGauss; k++)
                {
                    currDiffB = valB - ptrCurrModel[k].meanB;
                    currDiffG = valG - ptrCurrModel[k].meanG;
                    currDiffR = valR - ptrCurrModel[k].meanR;
#if CMPL_MOG_USE_APPROX_THRESHOLD
                    currDist = (currDiffB * currDiffB + currDiffG * currDiffG + currDiffR * currDiffR) / 
                               (ptrCurrModel[k].varB + ptrCurrModel[k].varG + ptrCurrModel[k].varR);
#else
                    currDist = currDiffB * currDiffB / ptrCurrModel[k].varB + 
                               currDiffG * currDiffG / ptrCurrModel[k].varG + 
                               currDiffR * currDiffR / ptrCurrModel[k].varR;
#endif
                    if (currDist < thresSqrMahaDist)
                    {
                        index = k;
                        break;
                    }
                }
                if (!ptrMask[j])
                    continue;
                if (index != -1)
                {
                    for (int k = 0; k < numGauss; k++)
                    {
                        if (index != k)
                            ptrCurrModel[k].weight -= learnRate * ptrCurrModel[k].weight;
                        else
                        {
                            ptrCurrModel[k].meanB += learnRate * currDiffB;
                            ptrCurrModel[k].meanG += learnRate * currDiffG;
                            ptrCurrModel[k].meanR += learnRate * currDiffR;
                            ptrCurrModel[k].varB += learnRate * (currDiffB * currDiffB - ptrCurrModel[k].varB);
                            ptrCurrModel[k].varG += learnRate * (currDiffG * currDiffG - ptrCurrModel[k].varG);
                            ptrCurrModel[k].varR += learnRate * (currDiffR * currDiffR - ptrCurrModel[k].varR);
                            if (ptrCurrModel[k].varB < minGaussVar) 
                                ptrCurrModel[k].varB = minGaussVar; 
                            if (ptrCurrModel[k].varG < minGaussVar) 
                                ptrCurrModel[k].varG = minGaussVar; 
                            if (ptrCurrModel[k].varR < minGaussVar) 
                                ptrCurrModel[k].varR = minGaussVar; 
                            ptrCurrModel[k].weight += learnRate * (1.0F - ptrCurrModel[k].weight);
                        }
                    }
                    for (int k = index -1; k >= 0; k--)
                    {
                        if (ptrCurrModel[k].weight < ptrCurrModel[k + 1].weight)
                        {
                            index = k;
                            ModelC3 temp = ptrCurrModel[k];
                            ptrCurrModel[k] = ptrCurrModel[k + 1];
                            ptrCurrModel[k + 1] = temp; 
                        }
                    }
                    float weightSum = 0;
                    for (int k = 0; k < numGauss; k++)
                        weightSum += ptrCurrModel[k].weight;
                    if (weightSum > 1.0001F || weightSum < 0.9999F)
                    {
                        weightSum = 1.0F / weightSum;
                        for (int k = 0; k < numGauss; k++)
                            ptrCurrModel[k].weight *= weightSum;
                    }
                }
                else
                {
                    ptrCurrModel[numGauss - 1].meanB = valB;
                    ptrCurrModel[numGauss - 1].meanG = valG;
                    ptrCurrModel[numGauss - 1].meanR = valR;
                    ptrCurrModel[numGauss - 1].varB = initGaussVar;
                    ptrCurrModel[numGauss - 1].varG = initGaussVar;
                    ptrCurrModel[numGauss - 1].varR = initGaussVar;
                    ptrCurrModel[numGauss - 1].weight = initWeight;
                    float weightSum = 0;
                    for (int k = 0; k < numGauss; k++)
                        weightSum += ptrCurrModel[k].weight;
                    weightSum = 1.0F / weightSum;
                    for (int k = 0; k < numGauss; k++)
                        ptrCurrModel[k].weight *= weightSum;
                    for (int k = numGauss - 2; k >= 0; k--)
                    {
                        if (ptrCurrModel[k].weight < ptrCurrModel[k + 1].weight)
                        {
                            index = k;
                            ModelC3 temp = ptrCurrModel[k];
                            ptrCurrModel[k] = ptrCurrModel[k + 1];
                            ptrCurrModel[k + 1] = temp; 
                        }
                    }
                }
            }
        }
    }
}

void zsfo::Mog::getBackground(Mat& back) const
{
    if (!model.data)
        THROW_EXCEPT("model is empty");
    back.create(height, width, type);
    if (type == CV_8UC1)
    {
        for (int i = 0; i < height; i++)
        {
            const ModelC1* ptrModel = (ModelC1*)model.ptr<unsigned char>(i);
            unsigned char* ptrBack = back.ptr<unsigned char>(i);
            for (int j = 0; j < width; j++)
                ptrBack[j] = saturate_cast<unsigned char>((ptrModel + j * numGauss)->mean);
        }
    }
    else
    {
        for (int i = 0; i < height; i++)
        {
            const ModelC3* ptrModel = (ModelC3*)model.ptr<unsigned char>(i);
            unsigned char* ptrBack = back.ptr<unsigned char>(i);
            for (int j = 0; j < width; j++)
            {
                const ModelC3* ptrCurrModel = ptrModel + j * numGauss;
                ptrBack[j * 3] = saturate_cast<unsigned char>(ptrCurrModel->meanB);
                ptrBack[j * 3 + 1] = saturate_cast<unsigned char>(ptrCurrModel->meanG);
                ptrBack[j * 3 + 2] = saturate_cast<unsigned char>(ptrCurrModel->meanR);
            }
        }
    }
}