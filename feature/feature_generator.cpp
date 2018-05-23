
#include "feature_generator.h"
#include <fstream>
#include <iostream>
#include <float.h>


using std::vector;

namespace apollo {
namespace perception {
namespace cnnseg {

bool FeatureGenerator::Init() {

  // raw feature parameters 
  range_ = 60;
  width_ = 640;
  height_ = 640;
  min_height_ = -5.0;
  max_height_ = 5.0;
  channels = 8;

  log_table_.resize(256);
  for (size_t i = 0; i < log_table_.size(); ++i) {
    log_table_[i] = std::log1p(static_cast<float>(1+i));
  }

  int siz = height_ * width_;

  // compute direction and distance features
  direction_data_.resize(siz, 0);
  distance_data_.resize(siz, 0);
  

  for (int row = 0; row < height_; ++row) {
    for (int col = 0; col < width_; ++col) {
      int idx = row * width_ + col;
      // * row <-> x, column <-> y
      float center_x = Pixel2Pc(row, height_, range_);
      float center_y = Pixel2Pc(col, width_, range_);
      const double K_CV_PI = 3.1415926535897932384626433832795;
      direction_data_[idx] =
              static_cast<float>(std::atan2(center_y, center_x) / (2.0 * K_CV_PI));
      distance_data_[idx] =
              static_cast<float>(std::hypot(center_x, center_y) / 60.0 - 0.5);
    }
  }
}


void FeatureGenerator::Generate(float* points,const int length, float* feature) {


  int siz = height_ * width_;

  //clear vector && compute six channel feature
  max_height_data_.resize(siz, -5);
  mean_height_data_.resize(siz, 0);
  count_data_.resize(siz, 0);
  top_intensity_data_.resize(siz, 0);
  mean_intensity_data_.resize(siz, 0);
  nonempty_data_.resize(siz, 0);

  map_idx_.resize(length/4);
  float inv_res_x = 0.5 * static_cast<float>(width_) / static_cast<float>(range_);
  float inv_res_y = 0.5 * static_cast<float>(height_) / static_cast<float>(range_);

  for (size_t i = 0; i < length; i+=4) {
    if (points[i+2] <= min_height_ || points[i+2] >= max_height_) {
      map_idx_[i/4] = -1;
      continue;
    }
    // * the coordinates of x and y are exchanged here
    // (row <-> x, column <-> y)
    int pos_x = F2I(points[i+1], range_, inv_res_x);  // col
    int pos_y = F2I(points[i], range_, inv_res_y);  // row
    if (pos_x >= width_ || pos_x < 0 || pos_y >= height_ || pos_y < 0) {
      map_idx_[i/4] = -1;
      continue;
    }
    map_idx_[i/4] = pos_y * width_ + pos_x;

    int idx = map_idx_[i/4];
    float pz = points[i+2];
    float pi = points[i+3] / 255.0;
    if (max_height_data_[idx] < pz) {
      max_height_data_[idx] = pz;
      top_intensity_data_[idx] = pi;
    }
    mean_height_data_[idx] += pz;
    mean_intensity_data_[idx] += pi;
    count_data_[idx] += 1.;
  }

  for (int i = 0; i < siz; ++i) {
    const double EPS = 1e-6;
    if (count_data_[i] < EPS) {
      max_height_data_[i] = 0;

    } else {
      mean_height_data_[i] /= count_data_[i];

      mean_intensity_data_[i] /= count_data_[i];

      nonempty_data_[i] = 1.;

    }
    count_data_[i] = LogCount(static_cast<int>(count_data_[i]));
  }

  //return float* feature
//  float max_0 = -FLT_MAX;
//  float max_1 = -FLT_MAX;
//  float max_2 = -FLT_MAX;
//  float max_3 = -FLT_MAX;
//  float max_4 = -FLT_MAX;
//  float max_5 = -FLT_MAX;
//  float max_6 = -FLT_MAX;
//  float max_7 = -FLT_MAX;

  for(size_t i=0; i<siz; ++i) {
    feature[i] = max_height_data_[i];
    feature[i+siz] = mean_height_data_[i];
    feature[i+siz*2] = count_data_[i];
    feature[i+siz*3] = direction_data_[i];
    feature[i+siz*4] = top_intensity_data_[i];
    feature[i+siz*5] = mean_intensity_data_[i];
    feature[i+siz*6] = distance_data_[i];
    feature[i+siz*7] = nonempty_data_[i];
  }

  /*
  // for debug
  for(size_t i=0; i<siz*channels; ++i)
  {
    pts["data"].append(Json::Value(feature[i]));
  }
    root["feature"].append(pts);
  std::ofstream outputfile;
  outputfile.open("/home/bai/Project/3D-Perpection/feature/test/now-0000000010.json");
  if(outputfile.is_open()) outputfile << writer.write(root);
  outputfile.close();
  printf("writer done!");
  */
}

}  // namespace cnnseg
}  // namespace perception
}  // namespace apollo
