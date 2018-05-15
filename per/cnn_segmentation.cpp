
#include "cnn_segmentation.h"
#include <jsoncpp/json/json.h>
#include <fstream>
#include <string>
#include <unistd.h>
#include <float.h>

namespace apollo {
namespace perception {

;
using std::string;
using std::vector;

bool CNNSegmentation::Init() {
  char* pwd = NULL;
  pwd = getcwd(NULL, 0);
  std::string root(pwd);
  std::string proto_file = root + "/../../dataset/deploy.prototxt";
  std::string weight_file = root + "/../../dataset/deploy.caffemodel";

  range_ = 60;
  width_ = 640;
  height_ = 640;

  float inv_res_x = 0.5 * static_cast<float>(width_) / range_;
  float inv_res_y = 0.5 * static_cast<float>(height_) / range_;

  /// Instantiate Caffe net
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
  caffe_net_.reset(new caffe::Net<float>(proto_file, caffe::TEST));
  caffe_net_->CopyTrainedLayersFrom(weight_file);
  
  std::cout << "using Caffe CPU mode";

  /// set related Caffe blobs
  // center offset prediction
  string instance_pt_blob_name =  "instance_pt";
  instance_pt_blob_ = caffe_net_->blob_by_name(instance_pt_blob_name);
  CHECK(instance_pt_blob_ != nullptr) << "`" << instance_pt_blob_name
                                      << "` not exists!";
  // objectness prediction
  string category_pt_blob_name = "category_score";
  category_pt_blob_ = caffe_net_->blob_by_name(category_pt_blob_name);
  CHECK(category_pt_blob_ != nullptr) << "`" << category_pt_blob_name
                                      << "` not exists!";
  // positiveness (foreground object probability) prediction
  string confidence_pt_blob_name = "confidence_score";
  confidence_pt_blob_ = caffe_net_->blob_by_name(confidence_pt_blob_name);
  CHECK(confidence_pt_blob_ != nullptr) << "`" << confidence_pt_blob_name
                                        << "` not exists!";
  // object height prediction
  string height_pt_blob_name = "height_pt";
  height_pt_blob_ = caffe_net_->blob_by_name(height_pt_blob_name);
  CHECK(height_pt_blob_ != nullptr) << "`" << height_pt_blob_name
                                    << "` not exists!";
  // raw feature data
  string feature_blob_name = "data";
  feature_blob_ = caffe_net_->blob_by_name(feature_blob_name);
  CHECK(feature_blob_ != nullptr) << "`" << feature_blob_name
                                  << "` not exists!";
  // class prediction
  string class_pt_blob_name = "class_score";
  class_pt_blob_ = caffe_net_->blob_by_name(class_pt_blob_name);
  CHECK(class_pt_blob_ != nullptr) << "`" << class_pt_blob_name
                                   << "` not exists!";

  cluster2d_.reset(new cnnseg::Cluster2D());
  if (!cluster2d_->Init(height_, width_, range_)) {
    std::cout << "Fail to Init cluster2d for CNNSegmentation";
  }

  feature_generator_.reset(new cnnseg::FeatureGenerator<float>());
  if (!feature_generator_->Init(feature_blob_.get())) {
    std::cout << "Fail to Init feature generator for CNNSegmentation" << std::endl;
    return false;
  }
  free(pwd);
  return true;
}

bool CNNSegmentation::Segment(const pcl_util::PointCloudPtr& pc_ptr,
                              const pcl_util::PointIndices& valid_indices,
                              vector<ObjectPtr>* objects) {
  objects->clear();
  int num_pts = static_cast<int>(pc_ptr->points.size());
  if (num_pts == 0) {
    std::cout << "None of input points, return directly.";
    return true;
  }

  use_full_cloud_ = false;

  // generate raw features
  feature_generator_->Generate(pc_ptr);
  // network forward process
  std::cout << "start forward" << std::endl;
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
  caffe_net_->Forward();
  std::cout << "forward done"<<std::endl;
  //PERF_BLOCK_END("[CNNSeg] CNN forward");

  // clutser points and construct segments/objects
  float objectness_thresh = 0.5;
  bool use_all_grids_for_clustering = false;
  cluster2d_->Cluster(*category_pt_blob_, *instance_pt_blob_, pc_ptr,
                      valid_indices, objectness_thresh,
                      use_all_grids_for_clustering);
  //PERF_BLOCK_END("[CNNSeg] clustering");

  cluster2d_->Filter(*confidence_pt_blob_, *height_pt_blob_);

  cluster2d_->Classify(*class_pt_blob_);

  float confidence_thresh = 0.1;
  float height_thresh = 0.5;
  int min_pts_num = 3;
  cluster2d_->GetObjects(confidence_thresh, height_thresh, min_pts_num,
                         objects);
  //PERF_BLOCK_END("[CNNSeg] post-processing");


  return true;
}

void CNNSegmentation::Preparefortracking(const std::vector<apollo::perception::ObjectPtr> &objects, string &fream_id, uint64_t &stamp)
{
  for(size_t i=0; i<objects.size(); ++i)
  {
    const ObjectPtr &obj = objects[i];
    CHECK_GT(obj->cloud->size(), 0);

    float x_min = FLT_MAX;
    float y_min = FLT_MAX;
    float z_min = FLT_MAX;
    float x_max = FLT_MIN;
    float y_max = FLT_MIN;
    float z_max = FLT_MIN;

    for (size_t j = 0; j < obj->cloud->size(); ++j) {
      const auto &point = obj->cloud->points[j];
      int col = F2I(point.y, range_, inv_res_x);  // col
      int row = F2I(point.x, range_, inv_res_y);  // row
      if(!IsValidRowCol(row, height_, col, width_)) continue;
      x_min = x_min > point.x ? point.x : x_min;
      x_max = x_max < point.x ? point.x : x_max;

      y_min = y_min > point.y ? point.y : y_min;
      y_max = y_max < point.y ? point.y : y_max;


      z_min = z_min > point.z ? point.z : z_min;
      z_max = z_max < point.z ? point.z : z_max;
    }

    obj->height = static_cast<double>(z_max - z_min);
    obj->width = static_cast<double>(y_max - y_min);
    obj->length = static_cast<double>(x_max - x_min);
    obj->stamp = stamp;
    obj->frame_id = fream_id;
//    std::cout << "height: " << obj->height << " " << "width: " << obj->width << " " << "length: " << obj->length << std::endl;
//    std::cout << "stamp: " << obj->stamp << " " << "frame_id: " << obj->frame_id << std::endl;
    double center_x = static_cast<double>((x_max - x_min) / 2);
    double center_y = static_cast<double>((y_max - y_min) / 2);
    double center_z = static_cast<double>((z_max - z_min) / 2);
    Eigen::Vector3d center = Eigen::Vector3d(center_x, center_y, center_z);
    obj->center = center;
//    std::cout << "center: " << obj->center << std::endl;
    //Follow the order of the requirements. counterclockwise
    obj->vertices.push_back(Eigen::Vector3d(static_cast<double>(x_max), static_cast<double>(y_max), static_cast<double>(z_max)));
    obj->vertices.push_back(Eigen::Vector3d(static_cast<double>(x_min), static_cast<double>(y_max), static_cast<double>(z_max)));
    obj->vertices.push_back(Eigen::Vector3d(static_cast<double>(x_min), static_cast<double>(y_min), static_cast<double>(z_max)));
    obj->vertices.push_back(Eigen::Vector3d(static_cast<double>(x_max), static_cast<double>(y_min), static_cast<double>(z_max)));
    obj->vertices.push_back(Eigen::Vector3d(static_cast<double>(x_max), static_cast<double>(y_max), static_cast<double>(z_min)));
    obj->vertices.push_back(Eigen::Vector3d(static_cast<double>(x_min), static_cast<double>(y_max), static_cast<double>(z_min)));
    obj->vertices.push_back(Eigen::Vector3d(static_cast<double>(x_min), static_cast<double>(y_min), static_cast<double>(z_min)));
    obj->vertices.push_back(Eigen::Vector3d(static_cast<double>(x_max), static_cast<double>(y_min), static_cast<double>(z_min)));
//    for(int i=0; i<8; ++i) std::cout << "vectices: " << obj->vertices[i] << std::endl;
  }
}


void CNNSegmentation::Write2Json(const std::vector<ObjectPtr> &objects , const std::string &json_path)
{
  std::ofstream outputfile;
  std::cout << "binaryfile: " << json_path << std::endl;
  
  //write to Json file
  
  Json::Value root;
  Json::Value temp;
  Json::Value pos;
  Json::StyledWriter writer;

  for(size_t i=0; i<objects.size(); ++i)
    {
        
        const ObjectPtr &obj = objects[i];
        temp["name"] = Json::Value(obj->type_name);
        temp["score"] = Json::Value(obj->score);
        
        
        for(size_t j=0; j<obj->cloud->size(); ++j)
        {
          Json::Value po;
          const auto &point = obj->cloud->points[j];
          
          pos["x"] = Json::Value(point.x);
          pos["y"] = Json::Value(point.y);
          pos["z"] = Json::Value(point.z);
          
          temp["points"].append(pos);
        }
        root["object"].append(temp);
        temp.clear();
        std::cout << "cnn obs cloud size:" << static_cast<int>(obj->cloud->size()) << std::endl;

      /*
      const ObjectPtr &obj = objects[i];
      //char temp[11] = obj->type_name;
      //fout << "type-name: " << obj->type_name <<"\n";
      fout << "type-score: " << obj->score << '\n';
      fout << "cloud-points size: " << obj->cloud->size() << '\n';
      for(size_t j=0; j<obj->cloud->size(); ++j)
      {
        const auto &point = obj->cloud->points[j];

        fout << point.x << " " << point.y << " " << point.z << '\n';
      }
      */

    }
  
  outputfile.open(json_path);
  if(outputfile.is_open())
  {
    std::cout << "start open " << std::endl;
      outputfile << writer.write(root);
  }
  outputfile.close();
}
}  // namespace perception
}  // namespace apollo
