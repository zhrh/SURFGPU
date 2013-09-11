#include "getSurfFeaGPU.h"
#include "surflib.h"
#define SURF_DECRIPTOR_DIM 64
int ExtractSurfFeatureGPU(unsigned char* data, int nw, int nh, int *feature_num,float* features)
{
	IpVec ipts;
	IplImage *img_gray = cvCreateImage(cvSize(nw,nh),IPL_DEPTH_8U,1);
	IplImage *img = cvCreateImage(cvSize(nw,nh),IPL_DEPTH_32F,1);
	memcpy(img_gray->imageData,data,nw * nh * sizeof(unsigned char));
	cvConvertScale(img_gray,img,1.0 / 255,0);
    surfDetDes(img,ipts,true,5,4,2,0.0004f);
	int desc_num = ipts.size();
    int i =0;
    for(i = 0;i != desc_num;++i)
    {
		memcpy(features + i * SURF_DECRIPTOR_DIM,ipts[i].descriptor,SURF_DECRIPTOR_DIM * sizeof(float));
    }
	*feature_num = desc_num;
	cvReleaseImage(&img);
	cvReleaseImage(&img_gray);
    return 0;
}
