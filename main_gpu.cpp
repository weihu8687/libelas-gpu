
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dirent.h>
#include <boost/filesystem.hpp>

#include "elas.h"
#include "elas_gpu.h"
#include "image.h"

using namespace std;
using namespace cv;

// Enable profiling
#define PROFILE

typedef std::vector<std::string> stringvec_t;

void read_directory(const std::string& name, stringvec_t& v, std::function<bool(std::string)> check)
{
  DIR* dirp = opendir(name.c_str());

  struct dirent * dp;
  while ((dp = readdir(dirp)) != NULL) {
    if(!check(dp->d_name))
    {
      continue;
    }

    v.push_back(dp->d_name);
  }
  closedir(dirp);
}


bool patternMatching(std::string mainStr, std::string toMatch, std::string starts)
{
//  if(mainStr.find(toMatch) != std::string::npos &&
//      mainStr.find(starts) == 0)
  if(mainStr.find(toMatch) != std::string::npos)
    return true;
  else
    return false;
}


bool DirectoryExists(const char *pzPath) {
  if (pzPath == NULL) return false;

  DIR *pDir;
  bool bExists = false;

  pDir = opendir(pzPath);

  if (pDir != NULL) {
    bExists = true;
    (void)closedir(pDir);
  }

  return bExists;
}


bool CheckAndCreateDir(std::string outputFilePath) {
  if (false == DirectoryExists(outputFilePath.c_str())) {
    cout << "Fail to open the output folder, we will create it: "
         << outputFilePath << endl;

    const char *path = outputFilePath.c_str();

    boost::filesystem::path dir(path);

    if (boost::filesystem::create_directory(dir)) {
      cout << "Directory Created: " << outputFilePath << endl;
      return true;
    } else {
      cout << "Fail to create output folder, exit!" << endl;
      return false;
    }
  }
}

cv::Mat creatColorDisparity(cv::Mat input)
{
  cv::Mat disparity_color = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);

  for (int j = 0; j < input.rows; j++) {
    for (int i = 0; i < input.cols; i++) {
      unsigned char val = input.at<unsigned char>(j, i);
      unsigned char pixR, pixG, pixB;

      if (val == 0)
        pixR = pixG = pixB = 0;
      else {
        // hotter is clearer, cooler is farther
        pixB = 255 - val;
        pixG = val < 128 ? val * 2 : (unsigned char)((255 - val) * 2);
        pixR = val;
      }
      disparity_color.at<cv::Vec3b>(j, i) = cv::Vec3b(pixB, pixG, pixR);
    }
  }

  return disparity_color;
}

/**
 * Compute disparities of pgm image input pair file_1, file_2
 */
//void process (const char* file_1,const char* file_2) {
void process(Mat imgL, Mat imgR, int frame_idx) {

  string outFileName = "./result/" + std::to_string(frame_idx) + ".png";

  // get image width and height
  int32_t width  = imgL.cols;
  int32_t height = imgL.rows;

  // allocate memory for disparity images
  const int32_t dims[3] = {width,height,width}; // bytes per line = width
  float* D1_data = (float*)malloc(width*height*sizeof(float));
  float* D2_data = (float*)malloc(width*height*sizeof(float));

  // process
  Elas::parameters param;
  param.postprocess_only_left = false;
  //param.subsampling = true;
  ElasGPU elas(param);
  elas.process(imgL.data,imgR.data,D1_data,D2_data,dims);

  Mat dispL = Mat(height, width, CV_32F, D1_data);
  Mat dispR = Mat(height, width, CV_32F, D2_data);

  double minVal, maxVal;
  cv::Mat disp8U;
  minMaxLoc(dispL, &minVal, &maxVal);
  cout << "minVal " << minVal << " maxVal " << maxVal << endl;
  dispL.convertTo(disp8U, CV_8U, 255 / (maxVal - minVal));

  cv::Mat disp_color = creatColorDisparity(disp8U);
  //disp_color.copyTo(displayMat);

  imwrite(outFileName, disp_color);

  // free memory
  free(D1_data);
  free(D2_data);
}

int main(int argc, char** argv) {

  if(argc == 2 && !strcmp(argv[1], "-h"))
  {
    cout << endl;
    cout << argv[0] << "  PATH_TO_image0_and_image1" << endl;
    return -1;
  }

  // Startup the GPU device
  // https://devtalk.nvidia.com/default/topic/895513/cuda-programming-and-performance/cudamalloc-slow/post/4724457/#4724457

  start_gpu_device();  //cudaFree(0);

  stringvec_t allimages_l;
  stringvec_t allimages_r;
  int idx = 0;

  string img_folder = argv[1];
  string left_path = img_folder + "/left";
  string right_path = img_folder + "/right";

  read_directory(left_path, allimages_l, [](std::string name){
    return patternMatching(name, ".png", "000");
  });
  std::sort(allimages_l.begin(), allimages_l.end());

  read_directory(right_path, allimages_r, [](std::string name){
    return patternMatching(name, ".png", "000");
  });
  std::sort(allimages_r.begin(), allimages_r.end());

  string outputDir = "./result";
  CheckAndCreateDir(outputDir);

  // read left image
  for (auto it_l = allimages_l.begin(), it_r = allimages_r.begin();
       it_l < allimages_l.end(), it_r < allimages_r.end(); it_l++, it_r++) {

    string left_name = left_path + "/" + *it_l;
    string right_name = right_path + "/" + *it_r;

    string findex = (*it_l).substr(0, 9);
    string outputImg = outputDir + "/" + findex + ".jpg";

    cout << "left_name " << left_name << endl;
    cout << "right_name " << right_name << endl;

    Mat imgLeft = imread(left_name, CV_LOAD_IMAGE_COLOR);
    Mat imgRight = imread(right_name, CV_LOAD_IMAGE_COLOR);

    if (!imgLeft.data || !imgRight.data) {
      cout << "fail to read img" << endl;
      break;
    }

    Mat imgLGray, imgRGray;
    cvtColor(imgLeft, imgLGray, CV_BGR2GRAY);
    cvtColor(imgRight, imgRGray, CV_BGR2GRAY);

    Mat displayMat = Mat::zeros(imgLeft.rows * 2, imgLeft.cols, CV_8UC3);
    imgLeft.copyTo(displayMat(Range(0, imgLeft.rows), Range(0, imgLeft.cols)));

    Mat displayBottom = displayMat(Range(imgLeft.rows, imgLeft.rows * 2), Range(0, imgLeft.cols));

    process(std::move(imgLGray), std::move(imgRGray), idx);
    idx++;

//    // Process example frames
//    process("../input/cones_left.pgm", "../input/cones_right.pgm");
//    process("../input/aloe_left.pgm", "../input/aloe_right.pgm");
//    process("../input/raindeer_left.pgm", "../input/raindeer_right.pgm");
//    process("../input/urban1_left.pgm", "../input/urban1_right.pgm");
//    process("../input/urban2_left.pgm", "../input/urban2_right.pgm");
//    process("../input/urban3_left.pgm", "../input/urban3_right.pgm");
//    process("../input/urban4_left.pgm", "../input/urban4_right.pgm");
//    cout << "... done!" << endl;

  }
  // Done!
  return EXIT_SUCCESS;
}