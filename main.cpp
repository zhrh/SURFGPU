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

#include "surflib.h"
#include "kmeans.h"
#include "utils.h"
#include <ctime>
#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>

//-------------------------------------------------------
// In order to you use OpenSURF, the following illustrates
// some of the simple tasks you can do.  It takes only 1
// function call to extract described SURF features!
// Define PROCEDURE as:
//  - 1 and supply image path to run on static image
//  - 2 to capture from a webcam
//  - 3 to match find an object in an image (work in progress)
//  - 4 to display moving features (work in progress)
//  - 5 to show matches between static images
#define PROCEDURE 1
#define MAX_IMAGE_NUM 100000
int TraveDir(char *path,int *filenum,char *filepath);

int TraveDir(char *path,int *filenum,char *filepath)
{
  DIR * imgdb;
  struct dirent *file;
  struct stat st;
  imgdb = opendir(path);
  if(imgdb == NULL)
  {
    printf("Can not open the directory %s\n",path);
    return -1;
  }
  //static int filenum = 0;
  //printf("here 1.1\n");
  while((file = readdir(imgdb)))
  {
    if((strcmp(file->d_name,".") == 0) || (strcmp(file->d_name,"..") == 0))
      continue;
    
    char tmp_path[1024];
    memset(tmp_path,0,1024);
    sprintf(tmp_path,"%s/%s",path,file->d_name);
    printf("%s\n",tmp_path);
    if((stat(tmp_path,&st) >= 0) && S_ISDIR(st.st_mode))
    {
      TraveDir(tmp_path,filenum,filepath);
    }
    else
    {
      char *sub = strrchr(tmp_path,'.');
      if(strlen(sub) - 1 < 5)
      {
        char surfix[10];
        memset(surfix,0,10);
        strncpy(surfix,sub + 1, strlen(sub) - 1);
        printf("here 1.2 , surfix = %s\n",surfix);
        if (!strcmp(surfix,"jpg") || !strcmp(surfix,"JPG") || !strcmp(surfix,"jpeg") || !strcmp(surfix,"JPEG") 
            || !strcmp(surfix,"bmp") || !strcmp(surfix,"BMP") || !strcmp(surfix,"PNG") || !strcmp(surfix,"png") 
            || !strcmp(surfix,"pgm") || !strcmp(surfix,"ppm") || !strcmp(surfix,"TIF") || !strcmp(surfix,"tif") 
            || !strcmp(surfix,"tiff") || !strcmp(surfix,"TIF")) 
        {
          //printf("here 1.3\n");
          printf("tmp = %d\n",*filenum);
          memset(&filepath[(*filenum) * 1024],0,1024);
		  //printf("here 1.3.1\n");
          sprintf(&filepath[(*filenum) * 1024],"%s/%s",path,file->d_name);
          ++(*filenum);
        }
      }
    }
  }
  closedir(imgdb);
  return *filenum;
}
//-------------------------------------------------------

int mainImage(int argc, char **argv);
int mainVideo(void);
int mainMatch(void);
int mainMotionPoints(void);
int mainStaticMatch(void);
int mainKmeans(void);

//-------------------------------------------------------

int main(int argc, char **argv) 
{
  if (PROCEDURE == 1) return mainImage(argc,argv);
  if (PROCEDURE == 2) return mainVideo();
  if (PROCEDURE == 3) return mainMatch();
  if (PROCEDURE == 4) return mainMotionPoints();
  if (PROCEDURE == 5) return mainStaticMatch();
  if (PROCEDURE == 6) return mainKmeans();
}


//-------------------------------------------------------

int mainImage(int argc,char **argv)
{
  if(argc < 3)
  {
    printf("Para Error!\n");
    printf("Use: ImgPath, FeaPath\n");
    //return -1;
  }
  struct timeval start,end;
  int time_use;
  char *filepath = (char *)malloc(MAX_IMAGE_NUM * 1024 * sizeof(char));
  int imgnum = 0;
  imgnum = TraveDir(argv[1],&imgnum,filepath);
  printf("imgnum = %d\n",imgnum);
  float total_time = 0.0;
  int pic_num = 0;

 // gettimeofday(&start,NULL);
 // AllocateMem(640, 480);
 // gettimeofday(&end,NULL);
 // time_use = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
 // time_use /= 1000;
 // std::cout<< "Allocate Memory took: " <<time_use<< " ms" << std::endl;

  int i = 0;
  for(i = 0;i < imgnum;++i)
  {
    printf("%s\n",&filepath[i * 1024]);
    char imgfile[1024];
    memset(imgfile,0,1024);
    strncpy(imgfile,&filepath[i * 1024],1024);
	IpVec ipts;
	IplImage *img = cvLoadImage(imgfile);
    if(img == NULL)
      continue;
	int width = img->width;
    int height = img->height;
    if(width / 2 / 16 <= 0 || height / 2 / 16 <= 0) // 因为在建立生成尺度空间时,防止模板大于图像,2是int_sample,16是5octave的系数
      continue;
	if(width > 1000 || height > 1000)
		continue;
	//clock_t start = clock();
    gettimeofday(&start,NULL);
	surfDetDes(img, ipts, true, 5, 4, 2, 0.0004f);
	gettimeofday(&end,NULL);
	time_use = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
	//time_use /= 1000;
	//clock_t end = clock();
	++pic_num;
	//float time = float(end - start) / CLOCKS_PER_SEC * 1000;
	std::cout<< "OpenSURF found: " << ipts.size() << " interest points" << std::endl;
	std::cout<< "OpenSURF took: " <<time_use<< " us" << std::endl;
	total_time += time_use;
	printf("***********************Avarage Time: %f us\n",total_time / pic_num);      

	char feafile[1024];
	memset(feafile,0,1024);
	char *q = strrchr(imgfile,'/');
	char *sub = strrchr(q + 1,'.');
	char filename[1024];
	memset(filename,0,1024);
	strncpy(filename,q + 1, strlen(q) - 1 - strlen(sub));
	strcat(filename,".surf");
	sprintf(feafile,"%s/%s",argv[2],filename);
	FILE *pf_feature = fopen(feafile,"wb");
	if(pf_feature == NULL)
	{
	  printf("Can't Open the file:%s\n",feafile);
	  return -1;
	}
	int fea_num = ipts.size();
	fwrite(&fea_num,sizeof(int),1,pf_feature);
	int i = 0;
	for(i = 0;i < fea_num;++i)
	{
	  fwrite(ipts[i].descriptor,sizeof(float),64,pf_feature);
	}
	fclose(pf_feature);
	cvReleaseImage(&img);
  }
  printf("Feature Number = %d\n",pic_num);
  free(filepath);
  //ReleaseMem();

/*
  DIR * imgdb;
  struct dirent *p;
  char imgfile[1024];
  imgdb = opendir(argv[1]);
  if(imgdb == NULL)
  {
    printf("Can not open the image directory %s\n",argv[1]);
    return -1;
  }
  float total_time = 0.0;
  int pic_num = 0;
  while((p = readdir(imgdb)))
  {
    if((strcmp(p->d_name,".") == 0) || (strcmp(p->d_name,"..") == 0))
      continue;
	else
	{
      memset(imgfile,0,1024);
      sprintf(imgfile,"%s/%s",argv[1],p->d_name);
	  printf("%s\n",imgfile);
      IpVec ipts;
      IplImage *img = cvLoadImage(imgfile);
	  if(img == NULL)
		continue;
	  int width = img->width;
	  int height = img->height;
	  if(width / 2 / 16 <= 0 || height / 2 / 16 <= 0) // 因为在建立生成尺度空间时,防止模板大于图像,2是int_sample,16是5octave的系数
		continue;
	  gettimeofday(&start,NULL);
	  surfDetDes(img, ipts, true, 5, 4, 2, 0.0004f);
	  gettimeofday(&end,NULL);
	  time_use = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
	  time_use /= 1000;
	  ++pic_num;
      std::cout<< "OpenSURF found: " << ipts.size() << " interest points" << std::endl;
      std::cout<< "OpenSURF took: " <<time_use<< " ms" << std::endl;
      total_time += time_use;
      printf("***********************Avarage Time: %f ms\n",total_time / pic_num);      

	  //Draw the detected points
	  //drawIpoints(img, ipts);

	  //Display the result
	  //showImage(img);
      char feafile[1024];
      memset(feafile,0,1024);
      char *q = strrchr(imgfile,'/');
      char *sub = strrchr(q + 1,'.');
      char filename[1024];
      memset(filename,0,1024);
      strncpy(filename,q + 1, strlen(q) - 1 - strlen(sub));
      strcat(filename,".surf");
      sprintf(feafile,"%s/%s",argv[2],filename);
      FILE *pf_feature = fopen(feafile,"wb");
      if(pf_feature == NULL)
      {
        printf("Can't Open the file:%s\n",feafile);
        return -1;
      }
      int fea_num = ipts.size();
      fwrite(&fea_num,sizeof(int),1,pf_feature);
      int i = 0;
      for(i = 0;i < fea_num;++i)
      {
        fwrite(ipts[i].descriptor,sizeof(float),64,pf_feature);
      }
      fclose(pf_feature);
    }
    
  }
  closedir(imgdb);
*/
/*
  // Declare Ipoints and other stuff
  IpVec ipts;
  IplImage *img=cvLoadImage("Images/sf.jpg");

  struct timeval start,end;
  gettimeofday(&start,NULL);
  // Detect and describe interest points in the image
  {
    surfDetDes(img, ipts, false, 3, 4, 2, 0.0004f); 
  }
  gettimeofday(&end,NULL);
  int time_use = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
  time_use /= 1000;
  std::cout<<"Detector and Descriptor time is:"<<time_use<<"ms"<<std::endl;

  std::cout<< "OpenSURF found: " << ipts.size() << " interest points" << std::endl;
  //std::cout<< "OpenSURF took: min/avg/max/stddev " << time_min << "/" << time_avg << "/" << time_max << "/" << stddev
	//		<< std::endl;

  // Draw the detected points
  //drawIpoints(img, ipts);

  // Display the result
  //showImage(img);
*/
  return 0;
}


//-------------------------------------------------------


int mainVideo(void)
{
  // Initialise capture device
  CvCapture* capture = cvCaptureFromCAM( CV_CAP_ANY );
  if(!capture) error("No Capture");

  // Create a window 
  cvNamedWindow("OpenSURF", CV_WINDOW_AUTOSIZE );

  // Declare Ipoints and other stuff
  IpVec ipts;
  IplImage *img=NULL;

  // Main capture loop
  while( 1 ) 
  {
    // Grab frame from the capture source
    img = cvQueryFrame(capture);

    // Extract surf points
    surfDetDes(img, ipts, true, 3, 4, 2, 0.004f);    

    // Draw the detected points
    drawIpoints(img, ipts);

    // Draw the FPS figure
    drawFPS(img);

    // Display the result
    cvShowImage("OpenSURF", img);

    // If ESC key pressed exit loop
    if( (cvWaitKey(10) & 255) == 27 ) break;
  }

  cvReleaseCapture( &capture );
  cvDestroyWindow( "OpenSURF" );
  return 0;
}


//-------------------------------------------------------


int mainMatch(void)
{
  // Initialise capture device
  CvCapture* capture = cvCaptureFromCAM( CV_CAP_ANY );
  if(!capture) error("No Capture");

  // Declare Ipoints and other stuff
  IpPairVec matches;
  IpVec ipts, ref_ipts;
  
  // This is the reference object we wish to find in video frame
  // Replace the line below with IplImage *img = cvLoadImage("Images/object.jpg"); 
  // where object.jpg is the planar object to be located in the video
  IplImage *img = cvLoadImage("Images/object.jpg"); 
  if (img == NULL) error("Need to load reference image in order to run matching procedure");
  CvPoint src_corners[4] = {{0,0}, {img->width,0}, {img->width, img->height}, {0, img->height}};
  CvPoint dst_corners[4];

  // Extract reference object Ipoints
  surfDetDes(img, ref_ipts, false, 3, 4, 3, 0.004f);
  drawIpoints(img, ref_ipts);
  showImage(img);

  // Create a window 
  cvNamedWindow("OpenSURF", CV_WINDOW_AUTOSIZE );

  // Main capture loop
  while( true ) 
  {
    // Grab frame from the capture source
    img = cvQueryFrame(capture);
     
    // Detect and describe interest points in the frame
    surfDetDes(img, ipts, false, 3, 4, 3, 0.004f);

    // Fill match vector
    getMatches(ipts,ref_ipts,matches);
    
    // This call finds where the object corners should be in the frame
    if (translateCorners(matches, src_corners, dst_corners))
    {
      // Draw box around object
      for(int i = 0; i < 4; i++ )
      {
        CvPoint r1 = dst_corners[i%4];
        CvPoint r2 = dst_corners[(i+1)%4];
        cvLine( img, cvPoint(r1.x, r1.y),
          cvPoint(r2.x, r2.y), cvScalar(255,255,255), 3 );
      }

      for (unsigned int i = 0; i < matches.size(); ++i)
        drawIpoint(img, matches[i].first);
    }

    // Draw the FPS figure
    drawFPS(img);

    // Display the result
    cvShowImage("OpenSURF", img);

    // If ESC key pressed exit loop
    if( (cvWaitKey(10) & 255) == 27 ) break;
  }

  // Release the capture device
  cvReleaseCapture( &capture );
  cvDestroyWindow( "OpenSURF" );
  return 0;
}


//-------------------------------------------------------


int mainMotionPoints(void)
{
  // Initialise capture device
  CvCapture* capture = cvCaptureFromCAM( CV_CAP_ANY );
  if(!capture) error("No Capture");

  // Create a window 
  cvNamedWindow("OpenSURF", CV_WINDOW_AUTOSIZE );

  // Declare Ipoints and other stuff
  IpVec ipts, old_ipts, motion;
  IpPairVec matches;
  IplImage *img;

  // Main capture loop
  while( 1 ) 
  {
    // Grab frame from the capture source
    img = cvQueryFrame(capture);

    // Detect and describe interest points in the image
    old_ipts = ipts;
    surfDetDes(img, ipts, true, 3, 4, 2, 0.0004f);

    // Fill match vector
    getMatches(ipts,old_ipts,matches);
    for (unsigned int i = 0; i < matches.size(); ++i) 
    {
      const float & dx = matches[i].first.dx;
      const float & dy = matches[i].first.dy;
      float speed = sqrt(dx*dx+dy*dy);
      if (speed > 5 && speed < 30) 
        drawIpoint(img, matches[i].first, 3);
    }
        
    // Display the result
    cvShowImage("OpenSURF", img);

    // If ESC key pressed exit loop
    if( (cvWaitKey(10) & 255) == 27 ) break;
  }

  // Release the capture device
  cvReleaseCapture( &capture );
  cvDestroyWindow( "OpenSURF" );
  return 0;
}


//-------------------------------------------------------

int mainStaticMatch()
{
  IplImage *img1, *img2;
  img1 = cvLoadImage("Images/img1.jpg");
  img2 = cvLoadImage("Images/img2.jpg");

  IpVec ipts1, ipts2;
  surfDetDes(img1,ipts1,false,4,4,2,0.0001f);
  surfDetDes(img2,ipts2,false,4,4,2,0.0001f);

  IpPairVec matches;
  getMatches(ipts1,ipts2,matches);

  for (unsigned int i = 0; i < matches.size(); ++i)
  {
    drawPoint(img1,matches[i].first);
    drawPoint(img2,matches[i].second);
  
    const int & w = img1->width;
    cvLine(img1,cvPoint(matches[i].first.x,matches[i].first.y),cvPoint(matches[i].second.x+w,matches[i].second.y), cvScalar(255,255,255),1);
    cvLine(img2,cvPoint(matches[i].first.x-w,matches[i].first.y),cvPoint(matches[i].second.x,matches[i].second.y), cvScalar(255,255,255),1);
  }

  std::cout << "Matches: " << matches.size() << std::endl;

  cvNamedWindow("1", CV_WINDOW_AUTOSIZE );
  cvNamedWindow("2", CV_WINDOW_AUTOSIZE );
  cvShowImage("1", img1);
  cvShowImage("2", img2);
  cvWaitKey(0);

  return 0;
}

//-------------------------------------------------------

int mainKmeans(void)
{
  IplImage *img = cvLoadImage("Images/img1.jpg");
  IpVec ipts;
  Kmeans km;
  
  // Get Ipoints
  surfDetDes(img,ipts,true,3,4,2,0.0006f);

  for (int repeat = 0; repeat < 10; ++repeat)
  {

    IplImage *img = cvLoadImage("Images/img1.jpg");
    km.Run(&ipts, 5, true);
    drawPoints(img, km.clusters);

    for (unsigned int i = 0; i < ipts.size(); ++i)
    {
      cvLine(img, cvPoint(ipts[i].x,ipts[i].y), cvPoint(km.clusters[ipts[i].clusterIndex].x ,km.clusters[ipts[i].clusterIndex].y),cvScalar(255,255,255));
    }

    showImage(img);
  }

  return 0;
}
