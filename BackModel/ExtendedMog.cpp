#include "ExtendedMog.h"
#include "Exception.h"
using namespace std;
using namespace cv;

#define CMPL_MOG_USE_RELAXED_PARAM 1
#define CMPL_MOG_USE_APPROX_THRESHOLD 1

#if CMPL_MOG_USE_RELAXED_PARAM
static const int maxNumOfGauss = 4;
static const int maxCount = 1000;
static const float thresForeBack = 0.7;
static const float thresSqrMahaDist = 2.5 * 2.5;
static const float initGaussStdDev = 30;
static const float initGaussVar = 30 * 30;
static const float minGaussStdDev = 15;
static const float minGaussVar = 15 * 15;
static const float initWeight = 0.05;
#else
static const int maxNumOfGauss = 4;
static const int maxCount = 1000;
static const float thresForeBack = 0.7;
static const float thresSqrMahaDist = 2.5 * 2.5;
static const float initGaussStdDev = 15;
static const float initGaussVar = 15 * 15;
static const float minGaussStdDev = 8;
static const float minGaussVar = 8 * 8;
static const float initWeight = 0.05;
#endif
   
void zsfo::Mog::init(const Mat& image)
{
	if (image.rows <= 0 || image.cols <= 0 || image.type() != CV_8UC1 && image.type() != CV_8UC3)
        THROW_EXCEPT("input image format not supported");

    frameSize = image.size();
	frameRect = Rect(0, 0, frameSize.width, frameSize.height);
	frameType = image.type();
    frameCount = 0;
    
    // for each gaussian mixture of each pixel bg model we store ...
    // the mixture sort key (w/sum_of_variances), the mixture weight (w),
    // the mean (nchannels values) and
    // the diagonal covariance matrix (another nchannels values)
    bgmodel.create(1, frameSize.height * frameSize.width * maxNumOfGauss * (2 + 2 * image.channels()), CV_32F);
    bgmodel = Scalar::all(0);

	mask.create(frameSize, CV_8UC1);
}

void zsfo::Mog::reset(void)
{
	bgmodel = Scalar::all(0);
    frameCount = 0;
}

namespace
{
template<typename VT> struct MixData
{
    float sortKey;
    float weight;
    VT mean;
    VT var;
};
}
    
static void process8UC1(const Mat& image, const Mat& mask, Mat& fore, Mat& back, double learningRate, Mat& bgmodel)
{
    int x, y, k, k1, rows = image.rows, cols = image.cols;
    float alpha = (float)learningRate, T = (float)thresForeBack, vT = (float)thresSqrMahaDist;
    int K = maxNumOfGauss;
    MixData<float>* mptr = (MixData<float>*)bgmodel.data;
    
    const float w0 = (float)initWeight;
    const float sk0 = (float)(w0 / initGaussStdDev);
    const float var0 = (float)(initGaussVar);
    const float minVar = (float)(minGaussVar);
    
    for (y = 0; y < rows; y++)
    {
        const unsigned char* src = image.ptr<unsigned char>(y);
		const unsigned char* run = mask.ptr<unsigned char>(y);
        unsigned char* dstFore = fore.ptr<unsigned char>(y);
		unsigned char* dstBack = back.ptr<unsigned char>(y);
        
        for (x = 0; x < cols; x++, mptr += K)
        {
            if (run[x])
			{
				float wsum = 0;
				float pix = src[x];
				int kHit = -1, kForeground = -1;
                
				for (k = 0; k < K; k++)
				{
					float w = mptr[k].weight;
					wsum += w;
					if (w < FLT_EPSILON)
						break;
					float mu = mptr[k].mean;
					float var = mptr[k].var;
					float diff = pix - mu;
					float d2 = diff * diff;
					if (d2 < vT * var)
					{
						wsum -= w;
						float dw = alpha * (1.F - w);
						mptr[k].weight = w + dw;
						mptr[k].mean = mu + alpha * diff;
						var = max(var + alpha * (d2 - var), minVar);
						mptr[k].var = var;
						mptr[k].sortKey = w / sqrt(var);
                        
						for (k1 = k - 1; k1 >= 0; k1--)
						{
							if (mptr[k1].sortKey >= mptr[k1 + 1].sortKey)
								break;
							std::swap(mptr[k1], mptr[k1 + 1]);
						}
                        
						kHit = k1 + 1;
						break;
					}
				}
                
				if (kHit < 0) // no appropriate gaussian mixture found at all, remove the weakest mixture and create a new one
				{
					kHit = k = min(k, K - 1);
					wsum += w0 - mptr[k].weight;
					mptr[k].weight = w0;
					mptr[k].mean = pix;
					mptr[k].var = var0;
					mptr[k].sortKey = sk0;
				}
				else
					for (; k < K; k++)
						wsum += mptr[k].weight;
                
				float wscale = 1.F / wsum;
				wsum = 0;
				for (k = 0; k < K; k++)
				{
					wsum += mptr[k].weight *= wscale;
					mptr[k].sortKey *= wscale;
					if (wsum > T && kForeground < 0)
						kForeground = k+1;
				}
                
				dstFore[x] = (unsigned char)(kHit >= kForeground ? 255 : 0);
				dstBack[x] = saturate_cast<unsigned char>(mptr[0].mean);
			}
			else
			{
                float pix = src[x];
                int kHit = -1, kForeground = -1;
                
                for (k = 0; k < K; k++)
                {
                    if (mptr[k].weight < FLT_EPSILON)
                        break;
                    float mu = mptr[k].mean;
                    float var = mptr[k].var;
                    float diff = pix - mu;
                    float d2 = diff * diff;
                    if (d2 < vT * var)
                    {
                        kHit = k;
                        break;
                    }
                }
                
                if (kHit >= 0)
                {
                    float wsum = 0;
                    for (k = 0; k < K; k++)
                    {
                        wsum += mptr[k].weight;
                        if (wsum > T)
                        {
                            kForeground = k+1;
                            break;
                        }
                    }
                }
                
                dstFore[x] = (unsigned char)(kHit < 0 || kHit >= kForeground ? 255 : 0);
				dstBack[x] = saturate_cast<unsigned char>(mptr[0].mean);
            }
        }
    }

	//mptr = (MixData<float>*)bgmodel.data;
	//Mat* backImages = new Mat[K];
	//for (int i = 0; i < K; i++)
	//	backImages[i].create(rows, cols, CV_8UC1);
	//unsigned char** ptr = new unsigned char*[K];
	//for (int k = 0; k < K; k++)
	//	ptr[k] = backImages[k].data;
	//for (int i = 0; i < rows; i++)
	//{
	//	for (int j = 0; j < cols; j++)
	//	{
	//		for(int k = 0; k < K; k++)
	//		{
	//			ptr[k][0] = saturate_cast<unsigned char>(mptr[0].mean);
	//			ptr[k]++;
	//			mptr++;
	//		}
	//	}
	//}
	//for (int k = 0; k < K; k++)
	//{
	//	char name[100];
	//	sprintf(name, "back u8c1 %d", k + 1);
	//	imshow(name, backImages[k]);
	//}
	//delete [] ptr;
	//delete [] backImages;
}
    
static void process8UC3(const Mat& image, const Mat& mask, Mat& fore, Mat& back, double learningRate, Mat& bgmodel)
{
    int x, y, k, k1, rows = image.rows, cols = image.cols;
    float alpha = (float)learningRate, T = (float)thresForeBack, vT = (float)thresSqrMahaDist;
    int K = maxNumOfGauss;
    
    const float w0 = (float)initWeight;
    const float sk0 = (float)(w0 / (initGaussStdDev));
    const float var0 = (float)(initGaussVar);
    const float minVar = (float)(minGaussVar);
    MixData<Vec3f>* mptr = (MixData<Vec3f>*)bgmodel.data;
    
    for (y = 0; y < rows; y++)
    {
        const unsigned char* src = image.ptr<unsigned char>(y);
		const unsigned char* run = mask.ptr<unsigned char>(y);
        unsigned char* dstFore = fore.ptr<unsigned char>(y);
		unsigned char* dstBack = back.ptr<unsigned char>(y);
        
        for (x = 0; x < cols; x++, mptr += K)
        {
			if (run[x])
			{
                float wsum = 0;
                Vec3f pix(src[x * 3], src[x * 3 + 1], src[x * 3 + 2]);
                int kHit = -1, kForeground = -1;
                
                for (k = 0; k < K; k++)
                {
                    float w = mptr[k].weight;
                    wsum += w;
                    if (w < FLT_EPSILON)
                        break;
                    Vec3f mu = mptr[k].mean;
                    Vec3f var = mptr[k].var;
                    Vec3f diff = pix - mu;
                    float d2 = diff.dot(diff);
#if CMPL_MOG_USE_APPROX_THRESHOLD
                    if (d2 < vT * (var[0] + var[1] + var[2]))
#else
					if (diff[0] * diff[0] / var[0] + 
						diff[1] * diff[1] / var[1] + 
						diff[2] * diff[2] / var[2] < vT)
#endif
                    {
                        wsum -= w;
                        float dw = alpha * (1.F - w);
                        mptr[k].weight = w + dw;
                        mptr[k].mean = mu + alpha * diff;
                        var = Vec3f(max(var[0] + alpha * (diff[0] * diff[0] - var[0]), minVar),
                                    max(var[1] + alpha * (diff[1] * diff[1] - var[1]), minVar),
                                    max(var[2] + alpha * (diff[2] * diff[2] - var[2]), minVar));
                        mptr[k].var = var;
                        mptr[k].sortKey = w / sqrt(var[0] + var[1] + var[2]);
                        
                        for (k1 = k - 1; k1 >= 0; k1--)
                        {
                            if (mptr[k1].sortKey >= mptr[k1 + 1].sortKey)
                                break;
                            std::swap(mptr[k1], mptr[k1 + 1]);
                        }
                        
                        kHit = k1 + 1;
                        break;
                    }
                }
                
                if (kHit < 0) // no appropriate gaussian mixture found at all, remove the weakest mixture and create a new one
                {
                    kHit = k = min(k, K - 1);
                    wsum += w0 - mptr[k].weight;
                    mptr[k].weight = w0;
                    mptr[k].mean = pix;
                    mptr[k].var = Vec3f(var0, var0, var0);
                    mptr[k].sortKey = sk0;
                }
                else
                    for (; k < K; k++)
                        wsum += mptr[k].weight;
            
                float wscale = 1.f/wsum;
                wsum = 0;
                for (k = 0; k < K; k++)
                {
                    wsum += mptr[k].weight *= wscale;
                    mptr[k].sortKey *= wscale;
                    if (wsum > T && kForeground < 0)
                        kForeground = k+1;
                }

				//float testSum = src[x * 3] + src[x * 3 + 1] + src[x * 3 + 2];
				//float testRatioB = src[x * 3] / testSum;
				//float testRatioG = src[x * 3 + 1] / testSum;
				//float testRatioR = src[x * 3 + 2] / testSum;

				//float backSum = mptr[0].mean[0] + mptr[0].mean[1] + mptr[0].mean[2];
				//float backRatioB = mptr[0].mean[0] / backSum;
				//float backRatioG = mptr[0].mean[1] / backSum;
				//float backRatioR = mptr[0].mean[2] / backSum;

				//bool isBack = fabs(testRatioB - backRatioB) < 0.015F &&
				//	          fabs(testRatioG - backRatioG) < 0.015F &&
				//			  fabs(testRatioR - backRatioR) < 0.015F;

				//dstFore[x] = (unsigned char)((kHit >= kForeground && !isBack) ? 255 : 0);
				dstFore[x] = (unsigned char)(kHit >= kForeground ? 255 : 0);
				dstBack[x * 3] = saturate_cast<unsigned char>(mptr[0].mean[0]);
				dstBack[x * 3 + 1] = saturate_cast<unsigned char>(mptr[0].mean[1]);
				dstBack[x * 3 + 2] = saturate_cast<unsigned char>(mptr[0].mean[2]);
            }
			else
			{
                Vec3f pix(src[x * 3], src[x * 3 + 1], src[x * 3 + 2]);
                int kHit = -1, kForeground = -1;
                
                for (k = 0; k < K; k++)
                {
                    if (mptr[k].weight < FLT_EPSILON)
                        break;
                    Vec3f mu = mptr[k].mean;
                    Vec3f var = mptr[k].var;
                    Vec3f diff = pix - mu;
                    float d2 = diff.dot(diff);
#if CMPL_MOG_USE_APPROX_THRESHOLD
                    if (d2 < vT * (var[0] + var[1] + var[2]))
#else
					if (diff[0] * diff[0] / var[0] + 
						diff[1] * diff[1] / var[1] + 
						diff[2] * diff[2] / var[2] < vT)
#endif
                    {
                        kHit = k;
                        break;
                    }
                }
 
                if (kHit >= 0)
                {
                    float wsum = 0;
                    for (k = 0; k < K; k++)
                    {
                        wsum += mptr[k].weight;
                        if (wsum > T)
                        {
                            kForeground = k+1;
                            break;
                        }
                    }
                }

				//float testSum = src[x * 3] + src[x * 3 + 1] + src[x * 3 + 2];
				//float testRatioB = src[x * 3] / testSum;
				//float testRatioG = src[x * 3 + 1] / testSum;
				//float testRatioR = src[x * 3 + 2] / testSum;

				//float backSum = mptr[0].mean[0] + mptr[0].mean[1] + mptr[0].mean[2];
				//float backRatioB = mptr[0].mean[0] / backSum;
				//float backRatioG = mptr[0].mean[1] / backSum;
				//float backRatioR = mptr[0].mean[2] / backSum;

				//bool isBack = fabs(testRatioB - backRatioB) < 0.02 &&
				//	          fabs(testRatioG - backRatioG) < 0.02 &&
				//			  fabs(testRatioR - backRatioR) < 0.02;
                //           
                //dstFore[x] = (unsigned char)((kHit < 0 || kHit >= kForeground) && !isBack ? 255 : 0);
				dstFore[x] = (unsigned char)(kHit < 0 || kHit >= kForeground ? 255 : 0);
				dstBack[x * 3] = saturate_cast<unsigned char>(mptr[0].mean[0]);
				dstBack[x * 3 + 1] = saturate_cast<unsigned char>(mptr[0].mean[1]);
				dstBack[x * 3 + 2] = saturate_cast<unsigned char>(mptr[0].mean[2]);
            }
        }
    }

	//mptr = (MixData<Vec3f>*)bgmodel.data;
	//Mat* backImages = new Mat[K];
	//for (int i = 0; i < K; i++)
	//	backImages[i].create(rows, cols, CV_8UC3);
	//unsigned char** ptr = new unsigned char*[K];
	//for (int k = 0; k < K; k++)
	//	ptr[k] = backImages[k].data;
	//for (int i = 0; i < rows; i++)
	//{
	//	for (int j = 0; j < cols; j++)
	//	{
	//		for(int k = 0; k < K; k++)
	//		{
	//			ptr[k][0] = saturate_cast<unsigned char>(mptr[0].mean[0]);
	//			ptr[k][1] = saturate_cast<unsigned char>(mptr[0].mean[1]);
	//			ptr[k][2] = saturate_cast<unsigned char>(mptr[0].mean[2]);
	//			ptr[k] += 3;
	//			mptr++;
	//		}
	//	}
	//}
	//for (int k = 0; k < K; k++)
	//{
	//	char name[100];
	//	sprintf(name, "back u8c3 %d", k + 1);
	//	imshow(name, backImages[k]);
	//}
	//delete [] ptr;
	//delete [] backImages;
}

void zsfo::Mog::update(const Mat& image, Mat& foreImage, Mat& backImage, const vector<Rect>& noUpdate)
{
	if (image.size() != frameSize && image.type() != frameType)
        THROW_EXCEPT("input image size or format not supported");
    
    if (frameCount < maxCount)
		frameCount++;
    double learningRate = (frameCount <= 0) ? 1.0 : 1. / frameCount;

	mask.setTo(255);
	if (!noUpdate.empty())
	{
		for (int i = 0; i < noUpdate.size(); i++)
		{
			Mat temp = mask(noUpdate[i] & frameRect);
			temp.setTo(0);
		}
	}

	foreImage.create(frameSize, CV_8UC1);
	backImage.create(frameSize, frameType);
    if (frameType == CV_8UC1)
        process8UC1(image, mask, foreImage, backImage, learningRate, bgmodel);
    else if (frameType == CV_8UC3)
        process8UC3(image, mask, foreImage, backImage, learningRate, bgmodel);
}

static void process8UC1(const Mat& image, const Mat& mask, Mat& fore, double learningRate, Mat& bgmodel)
{
    int x, y, k, k1, rows = image.rows, cols = image.cols;
    float alpha = (float)learningRate, T = (float)thresForeBack, vT = (float)thresSqrMahaDist;
    int K = maxNumOfGauss;
    MixData<float>* mptr = (MixData<float>*)bgmodel.data;
    
    const float w0 = (float)initWeight;
    const float sk0 = (float)(w0 / initGaussStdDev);
    const float var0 = (float)(initGaussVar);
    const float minVar = (float)(minGaussVar);
    
    for (y = 0; y < rows; y++)
    {
        const unsigned char* src = image.ptr<unsigned char>(y);
		const unsigned char* run = mask.ptr<unsigned char>(y);
        unsigned char* dstFore = fore.ptr<unsigned char>(y);
        
        for (x = 0; x < cols; x++, mptr += K)
        {
            if (run[x])
			{
				float wsum = 0;
				float pix = src[x];
				int kHit = -1, kForeground = -1;
                
				for (k = 0; k < K; k++)
				{
					float w = mptr[k].weight;
					wsum += w;
					if (w < FLT_EPSILON)
						break;
					float mu = mptr[k].mean;
					float var = mptr[k].var;
					float diff = pix - mu;
					float d2 = diff * diff;
					if (d2 < vT * var)
					{
						wsum -= w;
						float dw = alpha * (1.F - w);
						mptr[k].weight = w + dw;
						mptr[k].mean = mu + alpha * diff;
						var = max(var + alpha * (d2 - var), minVar);
						mptr[k].var = var;
						mptr[k].sortKey = w / sqrt(var);
                        
						for (k1 = k - 1; k1 >= 0; k1--)
						{
							if (mptr[k1].sortKey >= mptr[k1 + 1].sortKey)
								break;
							std::swap(mptr[k1], mptr[k1 + 1]);
						}
                        
						kHit = k1 + 1;
						break;
					}
				}
                
				if (kHit < 0) // no appropriate gaussian mixture found at all, remove the weakest mixture and create a new one
				{
					kHit = k = min(k, K - 1);
					wsum += w0 - mptr[k].weight;
					mptr[k].weight = w0;
					mptr[k].mean = pix;
					mptr[k].var = var0;
					mptr[k].sortKey = sk0;
				}
				else
					for (; k < K; k++)
						wsum += mptr[k].weight;
                
				float wscale = 1.F / wsum;
				wsum = 0;
				for (k = 0; k < K; k++)
				{
					wsum += mptr[k].weight *= wscale;
					mptr[k].sortKey *= wscale;
					if (wsum > T && kForeground < 0)
						kForeground = k+1;
				}
                
				dstFore[x] = (unsigned char)(kHit >= kForeground ? 255 : 0);
			}
			else
			{
                float pix = src[x];
                int kHit = -1, kForeground = -1;
                
                for (k = 0; k < K; k++)
                {
                    if (mptr[k].weight < FLT_EPSILON)
                        break;
                    float mu = mptr[k].mean;
                    float var = mptr[k].var;
                    float diff = pix - mu;
                    float d2 = diff * diff;
                    if (d2 < vT * var)
                    {
                        kHit = k;
                        break;
                    }
                }
                
                if (kHit >= 0)
                {
                    float wsum = 0;
                    for (k = 0; k < K; k++)
                    {
                        wsum += mptr[k].weight;
                        if (wsum > T)
                        {
                            kForeground = k+1;
                            break;
                        }
                    }
                }
                
                dstFore[x] = (unsigned char)(kHit < 0 || kHit >= kForeground ? 255 : 0);
            }
        }
    }

	//mptr = (MixData<float>*)bgmodel.data;
	//Mat* backImages = new Mat[K];
	//for (int i = 0; i < K; i++)
	//	backImages[i].create(rows, cols, CV_8UC1);
	//unsigned char** ptr = new unsigned char*[K];
	//for (int k = 0; k < K; k++)
	//	ptr[k] = backImages[k].data;
	//for (int i = 0; i < rows; i++)
	//{
	//	for (int j = 0; j < cols; j++)
	//	{
	//		for(int k = 0; k < K; k++)
	//		{
	//			ptr[k][0] = saturate_cast<unsigned char>(mptr[0].mean);
	//			ptr[k]++;
	//			mptr++;
	//		}
	//	}
	//}
	//for (int k = 0; k < K; k++)
	//{
	//	char name[100];
	//	sprintf(name, "back u8c1 %d", k + 1);
	//	imshow(name, backImages[k]);
	//}
	//delete [] ptr;
	//delete [] backImages;
}
    
static void process8UC3(const Mat& image, const Mat& mask, Mat& fore, double learningRate, Mat& bgmodel)
{
    int x, y, k, k1, rows = image.rows, cols = image.cols;
    float alpha = (float)learningRate, T = (float)thresForeBack, vT = (float)thresSqrMahaDist;
    int K = maxNumOfGauss;
    
    const float w0 = (float)initWeight;
    const float sk0 = (float)(w0 / (initGaussStdDev));
    const float var0 = (float)(initGaussVar);
    const float minVar = (float)(minGaussVar);
    MixData<Vec3f>* mptr = (MixData<Vec3f>*)bgmodel.data;
    
    for (y = 0; y < rows; y++)
    {
        const unsigned char* src = image.ptr<unsigned char>(y);
		const unsigned char* run = mask.ptr<unsigned char>(y);
        unsigned char* dstFore = fore.ptr<unsigned char>(y);
        
        for (x = 0; x < cols; x++, mptr += K)
        {
			if (run[x])
			{
                float wsum = 0;
                Vec3f pix(src[x * 3], src[x * 3 + 1], src[x * 3 + 2]);
                int kHit = -1, kForeground = -1;
                
                for (k = 0; k < K; k++)
                {
                    float w = mptr[k].weight;
                    wsum += w;
                    if (w < FLT_EPSILON)
                        break;
                    Vec3f mu = mptr[k].mean;
                    Vec3f var = mptr[k].var;
                    Vec3f diff = pix - mu;
                    float d2 = diff.dot(diff);
#if CMPL_MOG_USE_APPROX_THRESHOLD
                    if (d2 < vT * (var[0] + var[1] + var[2]))
#else
					if (diff[0] * diff[0] / var[0] + 
						diff[1] * diff[1] / var[1] + 
						diff[2] * diff[2] / var[2] < vT)
#endif
                    {
                        wsum -= w;
                        float dw = alpha * (1.F - w);
                        mptr[k].weight = w + dw;
                        mptr[k].mean = mu + alpha * diff;
                        var = Vec3f(max(var[0] + alpha * (diff[0] * diff[0] - var[0]), minVar),
                                    max(var[1] + alpha * (diff[1] * diff[1] - var[1]), minVar),
                                    max(var[2] + alpha * (diff[2] * diff[2] - var[2]), minVar));
                        mptr[k].var = var;
                        mptr[k].sortKey = w / sqrt(var[0] + var[1] + var[2]);
                        
                        for (k1 = k - 1; k1 >= 0; k1--)
                        {
                            if (mptr[k1].sortKey >= mptr[k1 + 1].sortKey)
                                break;
                            std::swap(mptr[k1], mptr[k1 + 1]);
                        }
                        
                        kHit = k1 + 1;
                        break;
                    }
                }
                
                if (kHit < 0) // no appropriate gaussian mixture found at all, remove the weakest mixture and create a new one
                {
                    kHit = k = min(k, K - 1);
                    wsum += w0 - mptr[k].weight;
                    mptr[k].weight = w0;
                    mptr[k].mean = pix;
                    mptr[k].var = Vec3f(var0, var0, var0);
                    mptr[k].sortKey = sk0;
                }
                else
                    for (; k < K; k++)
                        wsum += mptr[k].weight;
            
                float wscale = 1.f/wsum;
                wsum = 0;
                for (k = 0; k < K; k++)
                {
                    wsum += mptr[k].weight *= wscale;
                    mptr[k].sortKey *= wscale;
                    if (wsum > T && kForeground < 0)
                        kForeground = k+1;
                }

				dstFore[x] = (unsigned char)(kHit >= kForeground ? 255 : 0);
            }
			else
			{
                Vec3f pix(src[x * 3], src[x * 3 + 1], src[x * 3 + 2]);
                int kHit = -1, kForeground = -1;
                
                for (k = 0; k < K; k++)
                {
                    if (mptr[k].weight < FLT_EPSILON)
                        break;
                    Vec3f mu = mptr[k].mean;
                    Vec3f var = mptr[k].var;
                    Vec3f diff = pix - mu;
                    float d2 = diff.dot(diff);
#if CMPL_MOG_USE_APPROX_THRESHOLD
                    if (d2 < vT * (var[0] + var[1] + var[2]))
#else
					if (diff[0] * diff[0] / var[0] + 
						diff[1] * diff[1] / var[1] + 
						diff[2] * diff[2] / var[2] < vT)
#endif
                    {
                        kHit = k;
                        break;
                    }
                }
 
                if (kHit >= 0)
                {
                    float wsum = 0;
                    for (k = 0; k < K; k++)
                    {
                        wsum += mptr[k].weight;
                        if (wsum > T)
                        {
                            kForeground = k+1;
                            break;
                        }
                    }
                }

				dstFore[x] = (unsigned char)(kHit < 0 || kHit >= kForeground ? 255 : 0);
            }
        }
    }

	//mptr = (MixData<Vec3f>*)bgmodel.data;
	//Mat* backImages = new Mat[K];
	//for (int i = 0; i < K; i++)
	//	backImages[i].create(rows, cols, CV_8UC3);
	//unsigned char** ptr = new unsigned char*[K];
	//for (int k = 0; k < K; k++)
	//	ptr[k] = backImages[k].data;
	//for (int i = 0; i < rows; i++)
	//{
	//	for (int j = 0; j < cols; j++)
	//	{
	//		for(int k = 0; k < K; k++)
	//		{
	//			ptr[k][0] = saturate_cast<unsigned char>(mptr[0].mean[0]);
	//			ptr[k][1] = saturate_cast<unsigned char>(mptr[0].mean[1]);
	//			ptr[k][2] = saturate_cast<unsigned char>(mptr[0].mean[2]);
	//			ptr[k] += 3;
	//			mptr++;
	//		}
	//	}
	//}
	//for (int k = 0; k < K; k++)
	//{
	//	char name[100];
	//	sprintf(name, "back u8c3 %d", k + 1);
	//	imshow(name, backImages[k]);
	//}
	//delete [] ptr;
	//delete [] backImages;
}

void zsfo::Mog::update(const Mat& image, Mat& foreImage, const vector<Rect>& noUpdate)
{
	if (image.size() != frameSize && image.type() != frameType)
        THROW_EXCEPT("input image size or format not supported");
    
    if (frameCount < maxCount)
		frameCount++;
    double learningRate = (frameCount <= 0) ? 1.0 : 1. / frameCount;

	mask.setTo(255);
	if (!noUpdate.empty())
	{
		for (int i = 0; i < noUpdate.size(); i++)
		{
			Mat temp = mask(noUpdate[i] & frameRect);
			temp.setTo(0);
		}
	}

	foreImage.create(frameSize, CV_8UC1);
    if (frameType == CV_8UC1)
        process8UC1(image, mask, foreImage, learningRate, bgmodel);
    else if (frameType == CV_8UC3)
        process8UC3(image, mask, foreImage, learningRate, bgmodel);
}

static void process8UC1(const Mat& image, const Mat& mask, double learningRate, Mat& bgmodel)
{
    int x, y, k, k1, rows = image.rows, cols = image.cols;
    float alpha = (float)learningRate, T = (float)thresForeBack, vT = (float)thresSqrMahaDist;
    int K = maxNumOfGauss;
    MixData<float>* mptr = (MixData<float>*)bgmodel.data;
    
    const float w0 = (float)initWeight;
    const float sk0 = (float)(w0 / initGaussStdDev);
    const float var0 = (float)(initGaussVar);
    const float minVar = (float)(minGaussVar);
    
    for (y = 0; y < rows; y++)
    {
        const unsigned char* src = image.ptr<unsigned char>(y);
		const unsigned char* run = mask.ptr<unsigned char>(y);
        
        for (x = 0; x < cols; x++, mptr += K)
        {
            if (run[x])
			{
				float wsum = 0;
				float pix = src[x];
				int kHit = -1, kForeground = -1;
                
				for (k = 0; k < K; k++)
				{
					float w = mptr[k].weight;
					wsum += w;
					if (w < FLT_EPSILON)
						break;
					float mu = mptr[k].mean;
					float var = mptr[k].var;
					float diff = pix - mu;
					float d2 = diff * diff;
					if (d2 < vT * var)
					{
						wsum -= w;
						float dw = alpha * (1.F - w);
						mptr[k].weight = w + dw;
						mptr[k].mean = mu + alpha * diff;
						var = max(var + alpha * (d2 - var), minVar);
						mptr[k].var = var;
						mptr[k].sortKey = w / sqrt(var);
                        
						for (k1 = k - 1; k1 >= 0; k1--)
						{
							if (mptr[k1].sortKey >= mptr[k1 + 1].sortKey)
								break;
							std::swap(mptr[k1], mptr[k1 + 1]);
						}
                        
						kHit = k1 + 1;
						break;
					}
				}
                
				if (kHit < 0) // no appropriate gaussian mixture found at all, remove the weakest mixture and create a new one
				{
					kHit = k = min(k, K - 1);
					wsum += w0 - mptr[k].weight;
					mptr[k].weight = w0;
					mptr[k].mean = pix;
					mptr[k].var = var0;
					mptr[k].sortKey = sk0;
				}
				else
					for (; k < K; k++)
						wsum += mptr[k].weight;
                
				float wscale = 1.F / wsum;
				wsum = 0;
				for (k = 0; k < K; k++)
				{
					wsum += mptr[k].weight *= wscale;
					mptr[k].sortKey *= wscale;
					if (wsum > T && kForeground < 0)
						kForeground = k+1;
				}
			}
			else
			{
                float pix = src[x];
                int kHit = -1, kForeground = -1;
                
                for (k = 0; k < K; k++)
                {
                    if (mptr[k].weight < FLT_EPSILON)
                        break;
                    float mu = mptr[k].mean;
                    float var = mptr[k].var;
                    float diff = pix - mu;
                    float d2 = diff * diff;
                    if (d2 < vT * var)
                    {
                        kHit = k;
                        break;
                    }
                }
                
                if (kHit >= 0)
                {
                    float wsum = 0;
                    for (k = 0; k < K; k++)
                    {
                        wsum += mptr[k].weight;
                        if (wsum > T)
                        {
                            kForeground = k+1;
                            break;
                        }
                    }
                }
            }
        }
    }

	//mptr = (MixData<float>*)bgmodel.data;
	//Mat* backImages = new Mat[K];
	//for (int i = 0; i < K; i++)
	//	backImages[i].create(rows, cols, CV_8UC1);
	//unsigned char** ptr = new unsigned char*[K];
	//for (int k = 0; k < K; k++)
	//	ptr[k] = backImages[k].data;
	//for (int i = 0; i < rows; i++)
	//{
	//	for (int j = 0; j < cols; j++)
	//	{
	//		for(int k = 0; k < K; k++)
	//		{
	//			ptr[k][0] = saturate_cast<unsigned char>(mptr[0].mean);
	//			ptr[k]++;
	//			mptr++;
	//		}
	//	}
	//}
	//for (int k = 0; k < K; k++)
	//{
	//	char name[100];
	//	sprintf(name, "back u8c1 %d", k + 1);
	//	imshow(name, backImages[k]);
	//}
	//delete [] ptr;
	//delete [] backImages;
}
    
static void process8UC3(const Mat& image, const Mat& mask, double learningRate, Mat& bgmodel)
{
    int x, y, k, k1, rows = image.rows, cols = image.cols;
    float alpha = (float)learningRate, T = (float)thresForeBack, vT = (float)thresSqrMahaDist;
    int K = maxNumOfGauss;
    
    const float w0 = (float)initWeight;
    const float sk0 = (float)(w0 / (initGaussStdDev));
    const float var0 = (float)(initGaussVar);
    const float minVar = (float)(minGaussVar);
    MixData<Vec3f>* mptr = (MixData<Vec3f>*)bgmodel.data;
    
    for (y = 0; y < rows; y++)
    {
        const unsigned char* src = image.ptr<unsigned char>(y);
		const unsigned char* run = mask.ptr<unsigned char>(y);
        
        for (x = 0; x < cols; x++, mptr += K)
        {
			if (run[x])
			{
                float wsum = 0;
                Vec3f pix(src[x * 3], src[x * 3 + 1], src[x * 3 + 2]);
                int kHit = -1, kForeground = -1;
                
                for (k = 0; k < K; k++)
                {
                    float w = mptr[k].weight;
                    wsum += w;
                    if (w < FLT_EPSILON)
                        break;
                    Vec3f mu = mptr[k].mean;
                    Vec3f var = mptr[k].var;
                    Vec3f diff = pix - mu;
                    float d2 = diff.dot(diff);
#if CMPL_MOG_USE_APPROX_THRESHOLD
                    if (d2 < vT * (var[0] + var[1] + var[2]))
#else
					if (diff[0] * diff[0] / var[0] + 
						diff[1] * diff[1] / var[1] + 
						diff[2] * diff[2] / var[2] < vT)
#endif
                    {
                        wsum -= w;
                        float dw = alpha * (1.F - w);
                        mptr[k].weight = w + dw;
                        mptr[k].mean = mu + alpha * diff;
                        var = Vec3f(max(var[0] + alpha * (diff[0] * diff[0] - var[0]), minVar),
                                    max(var[1] + alpha * (diff[1] * diff[1] - var[1]), minVar),
                                    max(var[2] + alpha * (diff[2] * diff[2] - var[2]), minVar));
                        mptr[k].var = var;
                        mptr[k].sortKey = w / sqrt(var[0] + var[1] + var[2]);
                        
                        for (k1 = k - 1; k1 >= 0; k1--)
                        {
                            if (mptr[k1].sortKey >= mptr[k1 + 1].sortKey)
                                break;
                            std::swap(mptr[k1], mptr[k1 + 1]);
                        }
                        
                        kHit = k1 + 1;
                        break;
                    }
                }
                
                if (kHit < 0) // no appropriate gaussian mixture found at all, remove the weakest mixture and create a new one
                {
                    kHit = k = min(k, K - 1);
                    wsum += w0 - mptr[k].weight;
                    mptr[k].weight = w0;
                    mptr[k].mean = pix;
                    mptr[k].var = Vec3f(var0, var0, var0);
                    mptr[k].sortKey = sk0;
                }
                else
                    for (; k < K; k++)
                        wsum += mptr[k].weight;
            
                float wscale = 1.f/wsum;
                wsum = 0;
                for (k = 0; k < K; k++)
                {
                    wsum += mptr[k].weight *= wscale;
                    mptr[k].sortKey *= wscale;
                    if (wsum > T && kForeground < 0)
                        kForeground = k+1;
                }
            }
			else
			{
                Vec3f pix(src[x * 3], src[x * 3 + 1], src[x * 3 + 2]);
                int kHit = -1, kForeground = -1;
                
                for (k = 0; k < K; k++)
                {
                    if (mptr[k].weight < FLT_EPSILON)
                        break;
                    Vec3f mu = mptr[k].mean;
                    Vec3f var = mptr[k].var;
                    Vec3f diff = pix - mu;
                    float d2 = diff.dot(diff);
#if CMPL_MOG_USE_APPROX_THRESHOLD
                    if (d2 < vT * (var[0] + var[1] + var[2]))
#else
					if (diff[0] * diff[0] / var[0] + 
						diff[1] * diff[1] / var[1] + 
						diff[2] * diff[2] / var[2] < vT)
#endif
                    {
                        kHit = k;
                        break;
                    }
                }
 
                if (kHit >= 0)
                {
                    float wsum = 0;
                    for (k = 0; k < K; k++)
                    {
                        wsum += mptr[k].weight;
                        if (wsum > T)
                        {
                            kForeground = k+1;
                            break;
                        }
                    }
                }
            }
        }
    }

	//mptr = (MixData<Vec3f>*)bgmodel.data;
	//Mat* backImages = new Mat[K];
	//for (int i = 0; i < K; i++)
	//	backImages[i].create(rows, cols, CV_8UC3);
	//unsigned char** ptr = new unsigned char*[K];
	//for (int k = 0; k < K; k++)
	//	ptr[k] = backImages[k].data;
	//for (int i = 0; i < rows; i++)
	//{
	//	for (int j = 0; j < cols; j++)
	//	{
	//		for(int k = 0; k < K; k++)
	//		{
	//			ptr[k][0] = saturate_cast<unsigned char>(mptr[0].mean[0]);
	//			ptr[k][1] = saturate_cast<unsigned char>(mptr[0].mean[1]);
	//			ptr[k][2] = saturate_cast<unsigned char>(mptr[0].mean[2]);
	//			ptr[k] += 3;
	//			mptr++;
	//		}
	//	}
	//}
	//for (int k = 0; k < K; k++)
	//{
	//	char name[100];
	//	sprintf(name, "back u8c3 %d", k + 1);
	//	imshow(name, backImages[k]);
	//}
	//delete [] ptr;
	//delete [] backImages;
}

void zsfo::Mog::update(const Mat& image, const vector<Rect>& noUpdate)
{
	if (image.size() != frameSize && image.type() != frameType)
        THROW_EXCEPT("input image size or format not supported");
    
    if (frameCount < maxCount)
		frameCount++;
    double learningRate = (frameCount <= 0) ? 1.0 : 1. / frameCount;

	mask.setTo(255);
	if (!noUpdate.empty())
	{
		for (int i = 0; i < noUpdate.size(); i++)
		{
			Mat temp = mask(noUpdate[i] & frameRect);
			temp.setTo(0);
		}
	}

    if (frameType == CV_8UC1)
        process8UC1(image, mask, learningRate, bgmodel);
    else if (frameType == CV_8UC3)
        process8UC3(image, mask, learningRate, bgmodel);
}

void zsfo::Mog::getBackground(Mat& backImage)
{
	if (frameType == CV_8UC1)
	{
		backImage.create(frameSize, CV_8UC1);
		MixData<float>* mptr = (MixData<float>*)bgmodel.data;
		unsigned char* ptr = backImage.data;
		int length = frameSize.height * frameSize.width;
		for (int i = 0; i < length; i++)
		{
			ptr[0] = saturate_cast<unsigned char>(mptr[0].mean);
			ptr += 1;
			mptr += maxNumOfGauss;
		}
	}
	else if (frameType == CV_8UC3)
	{
		backImage.create(frameSize, CV_8UC3);
		MixData<Vec3f>* mptr = (MixData<Vec3f>*)bgmodel.data;
		unsigned char* ptr = backImage.data;
		int length = frameSize.height * frameSize.width;
		for (int i = 0; i < length; i++)
		{
			ptr[0] = saturate_cast<unsigned char>(mptr[0].mean[0]);
			ptr[1] = saturate_cast<unsigned char>(mptr[0].mean[1]);
			ptr[2] = saturate_cast<unsigned char>(mptr[0].mean[2]);
			ptr += 3;
			mptr += maxNumOfGauss;
		}
	}
}