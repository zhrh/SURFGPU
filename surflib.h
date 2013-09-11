/*
 * Copyright (C) 2009-2010 Andre Schulz, Florian Jung, Sebastian Hartte,
 *						   Daniel Trick, Christan Wojek, Konrad Schindler,
 *						   Jens Ackermann, Michael Goesele
 * Copyright (C) 2008-2009 Christopher Evans <chris.evans@irisys.co.uk>, MSc University of Bristol
 *
 * This file is part of SURFGPU.
 *
 * SURFGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SURFGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SURFGPU.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SURFLIB_H
#define SURFLIB_H

#include "cv.h"
#include "highgui.h"
#include "integral.h"
#include "fasthessian.h"
#include "surf.h"
#include "ipoint.h"
#include "cudaimage.h"
#include <sys/time.h>
#include "global_variable.h"
#include <cuda_runtime.h>
#include "cuda/cudpp_helper_funcs.h"

/*
//! Check for CUDA error
#ifdef _DEBUG
#  define CUDA_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    err = CUDA_DEVICE_SYNCHRONIZE();                                           \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }
#else
#  define CUDA_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }
#endif
*/
/*
unsigned int *d_rgb_img = NULL;
float *d_gray_img = NULL;
float *d_int_img = NULL;
float *d_int_img_tr = NULL;
float *d_int_img_tr2 = NULL;
size_t rgb_img_pitch = 0, gray_img_pitch = 0, int_img_pitch = 0, int_img_tr_pitch = 0;
CUDPPHandle cudpp_lib;
CUDPPHandle mscan_plan, mscan_tr_plan;

inline void AllocateMem(int img_width, int img_height)
{
	
	CUDA_SAFE_CALL( cudaMallocPitch((void**)&d_rgb_img, &rgb_img_pitch, img_width * sizeof(unsigned int), img_height) );
	CUDA_SAFE_CALL( cudaMallocPitch((void**)&d_gray_img, &gray_img_pitch, img_width * sizeof(float), img_height) );
	CUDA_SAFE_CALL( cudaMallocPitch((void**)&d_int_img, &int_img_pitch, img_width * sizeof(float), img_height) );
	CUDA_SAFE_CALL( cudaMallocPitch((void**)&d_int_img_tr, &int_img_tr_pitch, img_height * sizeof(float), img_width) );
	CUDA_SAFE_CALL( cudaMallocPitch((void**)&d_int_img_tr2, &int_img_tr_pitch, img_height * sizeof(float), img_width) );

	CUDPP_SAFE_CALL( cudppCreate(&cudpp_lib) );

	CUDPPConfiguration cudpp_conf;
	cudpp_conf.op = CUDPP_ADD;
	cudpp_conf.datatype = CUDPP_FLOAT;
	cudpp_conf.algorithm = CUDPP_SCAN;
	cudpp_conf.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

	CUDPP_SAFE_CALL( cudppPlan(cudpp_lib, &mscan_plan, cudpp_conf,
		img_width, img_height, gray_img_pitch / sizeof(float)) );
	CUDPP_SAFE_CALL( cudppPlan(cudpp_lib, &mscan_tr_plan, cudpp_conf,
		img_height, img_width, int_img_tr_pitch / sizeof(float)) );
}

inline void ReleaseMem()
{
	CUDPP_SAFE_CALL( cudppDestroyPlan(mscan_plan) );
	CUDPP_SAFE_CALL( cudppDestroyPlan(mscan_tr_plan) );
	CUDPP_SAFE_CALL( cudppDestroy( cudpp_lib ) );
	CUDA_SAFE_CALL( cudaFree(d_rgb_img) );
	CUDA_SAFE_CALL( cudaFree(d_gray_img) );
	CUDA_SAFE_CALL( cudaFree(d_int_img_tr) );
	CUDA_SAFE_CALL( cudaFree(d_int_img_tr2) );
}
*/

//! Library function builds vector of described interest points
inline void surfDetDes(IplImage *img,  /* image to find Ipoints in */
                       std::vector<Ipoint> &ipts, /* reference to vector of Ipoints */
                       bool upright = false, /* run in rotation invariant mode? */
                       int octaves = OCTAVES, /* number of octaves to calculate */
                       int intervals = INTERVALS, /* number of intervals per octave */
                       int init_sample = INIT_SAMPLE, /* initial sampling step */
                       float thres = THRES /* blob response threshold */)
{
  
  struct timeval start,end;
  int time_use;
  gettimeofday(&start,NULL);
  // Create integral-image representation of the image
  cudaImage *int_img = Integral(img);
  gettimeofday(&end,NULL);
  time_use = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
  //time_use /= 1000;
  std::cout<<"Create integral-image time is:"<<time_use<<"us"<<std::endl;

  gettimeofday(&start,NULL);
  // Create Fast Hessian Object
  FastHessian fh(int_img, ipts, octaves, intervals, init_sample, thres);

  // Extract interest points and store in vector ipts
  fh.getIpoints();
  gettimeofday(&end,NULL);
  time_use = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
  //time_use /= 1000;
  std::cout<<"Extract interest points time is:"<<time_use<<"us"<<std::endl;

  gettimeofday(&start,NULL);
  if (ipts.size() > 0)
  {
    // Create Surf Descriptor Object
    Surf des(int_img, ipts);

    // Extract the descriptors for the ipts
    des.getDescriptors(upright);
  }
  gettimeofday(&end,NULL);
  time_use = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
  //time_use /= 1000;
  std::cout<<"Extarc descriptors time is:"<<time_use<<"us"<<std::endl;

  // Deallocate the integral image
  freeCudaImage(int_img);
  //cudaDeviceReset();
}


//! Library function builds vector of interest points
inline void surfDet(IplImage *img,  /* image to find Ipoints in */
                    std::vector<Ipoint> &ipts, /* reference to vector of Ipoints */
                    int octaves = OCTAVES, /* number of octaves to calculate */
                    int intervals = INTERVALS, /* number of intervals per octave */
                    int init_sample = INIT_SAMPLE, /* initial sampling step */
                    float thres = THRES /* blob response threshold */)
{
  // Create integral image representation of the image
  cudaImage *int_img = Integral(img);

  // Create Fast Hessian Object
  FastHessian fh(int_img, ipts, octaves, intervals, init_sample, thres);

  // Extract interest points and store in vector ipts
  fh.getIpoints();

  // Deallocate the integral image
  freeCudaImage(int_img);
}


//! Library function describes interest points in vector
inline void surfDes(IplImage *img,  /* image to find Ipoints in */
                    std::vector<Ipoint> &ipts, /* reference to vector of Ipoints */
                    bool upright = false) /* run in rotation invariant mode? */
{
  if (ipts.size() == 0) return;

  // Create integral image representation of the image
  cudaImage *int_img = Integral(img);

  // Create Surf Descriptor Object
  Surf des(int_img, ipts);

  // Extract the descriptors for the ipts
  des.getDescriptors(upright);

  // Deallocate the integral image
  freeCudaImage(int_img);
}

#endif /* SURFLIB_H */
