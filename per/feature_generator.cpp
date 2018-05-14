
#include "feature_generator.h"
#include "Config.h"
#include <fstream>
#include "util.h"

using std::vector;

namespace apollo {
namespace perception {
namespace cnnseg {

template <typename Dtype>
bool FeatureGenerator<Dtype>::Init(caffe::Blob<Dtype>* out_blob) {
  out_blob_ = out_blob;

  // raw feature parameters 
  range_ = 60;
  width_ = 640;
  height_ = 640;
  min_height_ = -5.0;
  max_height_ = 5.0;
  CHECK_EQ(width_, height_)
      << "Current implementation version requires input_width == input_height.";

  // set output blob and log lookup table
  out_blob_->Reshape(1, 8, height_, width_);

  log_table_.resize(256);
  for (size_t i = 0; i < log_table_.size(); ++i) {
    log_table_[i] = std::log1p(static_cast<Dtype>(1+i));
  }

  Dtype* out_blob_data = nullptr;
  out_blob_data = out_blob_->mutable_cpu_data();

  int channel_index = 0;
  max_height_data_ = out_blob_data + out_blob_->offset(0, channel_index++);
  mean_height_data_ = out_blob_data + out_blob_->offset(0, channel_index++);
  count_data_ = out_blob_data + out_blob_->offset(0, channel_index++);
  direction_data_ = out_blob_data + out_blob_->offset(0, channel_index++);
  top_intensity_data_ = out_blob_data + out_blob_->offset(0, channel_index++);
  mean_intensity_data_ = out_blob_data + out_blob_->offset(0, channel_index++);
  distance_data_ = out_blob_data + out_blob_->offset(0, channel_index++);
  nonempty_data_ = out_blob_data + out_blob_->offset(0, channel_index++);
  CHECK_EQ(out_blob_->offset(0, channel_index), out_blob_->count());

  // compute direction and distance features
  int siz = height_ * width_;
  vector<Dtype> direction_data( siz);
  vector<Dtype> distance_data(siz);
  
  

  for (int row = 0; row < height_; ++row) {
    for (int col = 0; col < width_; ++col) {
      int idx = row * width_ + col;
      // * row <-> x, column <-> y
      float center_x = Pixel2Pc(row, height_, range_);
      float center_y = Pixel2Pc(col, width_, range_);
      const double K_CV_PI = 3.1415926535897932384626433832795;
      direction_data[idx] =
          static_cast<Dtype>(std::atan2(center_y, center_x) / (2.0 * K_CV_PI));
      distance_data[idx] =
          static_cast<Dtype>(std::hypot(center_x, center_y) / 60.0 - 0.5);
    }
  }
  caffe::caffe_copy(siz, direction_data.data(), direction_data_);
  caffe::caffe_copy(siz, distance_data.data(), distance_data_);

  //input txt

  /*
  for (int i=0; i<siz; i++)
  {
      //pos["data"] = Json::Value(direction_data[i]);
      temp["data"].append(Json::Value(direction_data[i]));
  }
  feature["channel-3"].append(temp);
  temp.clear();

  for(int i=0; i<siz; ++i)
  {
      //pos["data"] = Json::Value(distance_data[i]);
      temp["data"].append(Json::Value(distance_data[i]));
  }
  feature["channel-6"].append(temp);
  temp.clear();
   */
}

template <typename Dtype>
void FeatureGenerator<Dtype>::Generate(
    const apollo::perception::pcl_util::PointCloudConstPtr& pc_ptr) {
  const auto& points = pc_ptr->points;
  // DO NOT remove this line!!!
  // Otherwise, the gpu_data will not be updated for the later frames.
  // It marks the head at cpu for blob.
  out_blob_->mutable_cpu_data();

  int siz = height_ * width_;
  caffe::caffe_set(siz, Dtype(-5), max_height_data_);
  caffe::caffe_set(siz, Dtype(0), mean_height_data_);
  caffe::caffe_set(siz, Dtype(0), count_data_);
  caffe::caffe_set(siz, Dtype(0), top_intensity_data_);
  caffe::caffe_set(siz, Dtype(0), mean_intensity_data_);
  caffe::caffe_set(siz, Dtype(0), nonempty_data_);

  map_idx_.resize(points.size());
  float inv_res_x = 0.5 * static_cast<float>(width_) / static_cast<float>(range_);
  float inv_res_y = 0.5 * static_cast<float>(height_) / static_cast<float>(range_);

  for (size_t i = 0; i < points.size(); ++i) {
    if (points[i].z <= min_height_ || points[i].z >= max_height_) {
      map_idx_[i] = -1;
      continue;
    }
    // * the coordinates of x and y are exchanged here
    // (row <-> x, column <-> y)
    int pos_x = F2I(points[i].y, range_, inv_res_x);  // col
    int pos_y = F2I(points[i].x, range_, inv_res_y);  // row
    if (pos_x >= width_ || pos_x < 0 || pos_y >= height_ || pos_y < 0) {
      map_idx_[i] = -1;
      continue;
    }
    map_idx_[i] = pos_y * width_ + pos_x;

    int idx = map_idx_[i];
    float pz = points[i].z;
    float pi = points[i].intensity / 255.0;
    if (max_height_data_[idx] < pz) {
      max_height_data_[idx] = pz;
      top_intensity_data_[idx] = pi;
    }
    mean_height_data_[idx] += static_cast<Dtype>(pz);
    mean_intensity_data_[idx] += static_cast<Dtype>(pi);
    count_data_[idx] += Dtype(1);
  }

  for (int i = 0; i < siz; ++i) {
    const double EPS = 1e-6;
    if (count_data_[i] < EPS) {
      max_height_data_[i] = Dtype(0);

    } else {
      mean_height_data_[i] /= count_data_[i];

      mean_intensity_data_[i] /= count_data_[i];

      nonempty_data_[i] = Dtype(1);

    }
    count_data_[i] = LogCount(static_cast<int>(count_data_[i]));
  }

    /*
    for(int i=0; i<siz; ++i)
    {
        temp["data"].append(Json::Value(max_height_data_[i]));
    }
    feature["channel-0"].append(temp);
    temp.clear();


    for(int i=0; i<siz; ++i)
    {
        temp["data"].append(mean_height_data_[i]);
    }
    feature["channel-1"].append(temp);
    temp.clear();


    for(int i=0; i<siz; ++i)
    {
        temp["data"].append(Json::Value(count_data_[i]));
    }
    feature["channel-2"].append(temp);
    temp.clear();


    for(int i=0; i<siz; ++i)
    {
        temp["data"].append(Json::Value(mean_intensity_data_[i]));
    }
    feature["channel-5"].append(temp);
    temp.clear();


    for(int i=0; i<siz; ++i)
    {
        temp["data"].append(Json::Value(nonempty_data_[i]));
    }
    feature["channel-7"].append(temp);
    temp.clear();


    for(int i=0; i<siz; ++i)
    {
        temp["data"].append(Json::Value(top_intensity_data_[i]));
    }
    feature["channel-4"].append(temp);
    temp.clear();

    root["feature"].append(feature);
    std::ofstream outputfile;
    outputfile.open("/home/bai/Project/cnn_seg/dataset/feature.json");
    if(outputfile.is_open())
    {
        std::cout << "start write feature to json " << std::endl;
        outputfile << writer.write(root);
    }
    outputfile.close();
     */
}

template bool FeatureGenerator<float>::Init(caffe::Blob<float>* blob);

template void FeatureGenerator<float>::Generate(
    const apollo::perception::pcl_util::PointCloudConstPtr& pc_ptr);

template bool FeatureGenerator<double>::Init(caffe::Blob<double>* blob);

template void FeatureGenerator<double>::Generate(
    const apollo::perception::pcl_util::PointCloudConstPtr& pc_ptr);

}  // namespace cnnseg
}  // namespace perception
}  // namespace apollo
