
#include "cnn_segmentation.h"

#include <jsoncpp/json/json.h>
#include<fstream>
#include <string>


namespace apollo {
namespace perception {

//using apollo::common::util::GetAbsolutePath;
using std::string;
using std::vector;

bool CNNSegmentation::Init() {
  
  std::string proto_file = PROTO_FILE;
  std::string weight_file = WEIGHT_FILE;
  
  /*
  if (!GetConfigs(&config_file, &proto_file, &weight_file)) {
    return false;
  }
  AINFO << "--    config_file: " << config_file;
  AINFO << "--     proto_file: " << proto_file;
  AINFO << "--    weight_file: " << weight_file;

  if (!apollo::common::util::GetProtoFromFile(config_file, &cnnseg_param_)) {
    AERROR << "Failed to load config file of CNNSegmentation.";
  }

  /// set parameters
  auto network_param = cnnseg_param_.network_param();
  auto feature_param = cnnseg_param_.feature_param();

  if (feature_param.has_point_cloud_range()) {
    range_ = static_cast<float>(feature_param.point_cloud_range());
  } else {
    range_ = 60.0;
  }
  if (feature_param.has_width()) {
    width_ = static_cast<int>(feature_param.width());
  } else {
    width_ = 640;
  }
  if (feature_param.has_height()) {
    height_ = static_cast<int>(feature_param.height());
  } else {
    height_ = 640;
  }
  */
  range_ = 60;
  width_ = 640;
  height_ = 640;
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

void CNNSegmentation::Write2Json(const std::vector<ObjectPtr> &objects , const std::string &json_path)
{
  std::ofstream outputfile;
  //std::string binaryfilepath = json_path;
  std::cout << "binaryfile: " << json_path << std::endl;
  
  //write to Json file
  Json::Value root;
  Json::Value temp;
  Json::Value pos;
  Json::StyledWriter writer;
  
  /*
  std::ofstream fout;
  fout.open(binaryfilepath, std::ios::out | std::ios::in | std::ios::app);
  fout << "Total-objects-size: " << objects.size() << '\n';
  */
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
      
          //std::cout << point.x << " " << point.y << " " << point.z << std::endl;
          
          temp["points"].append(pos);
        }
        root["object"].append(temp);
        temp.clear();
        std::cout << "cnn obs cloud size:" << static_cast<int>(obj->cloud->size()) << std::endl;
        //root["cloud"] = Json::Value(arrays);
        //delete[] arrays;
       
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
  

  //fout.close();
  std::cout << "end " << std::endl;
  
}
}  // namespace perception
}  // namespace apollo
