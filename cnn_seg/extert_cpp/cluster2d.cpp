

#ifndef CLUSTER2D_H_
#define CLUSTER2D_H_

#include <algorithm>
#include <vector>
#include <string>
#include "glog/logging.h"
#include "disjoint_set.h"
//#include "util.h"
#include <iostream>



#define AERROR LOG(ERROR)

namespace apollo {

inline int F2I(float val, float ori, float scale) {
    return static_cast<int>(std::floor((ori - val) * scale));
}

inline int Pc2Pixel(float in_pc, float in_range, float out_size) {
    float inv_res = 0.5 * out_size / in_range;
    return static_cast<int>(std::floor((in_range - in_pc) * inv_res));
}

inline float Pixel2Pc(int in_pixel, float in_size, float out_range) {
    float res = 2.0 * out_range / in_size;
    return out_range - (static_cast<float>(in_pixel) + 0.5f) * res;
}
using apollo::common::util::DisjointSetMakeSet;
using apollo::common::util::DisjointSetFind;
using apollo::common::util::DisjointSetUnion;

enum class MetaType {
  META_UNKNOWN,
  META_SMALLMOT,
  META_BIGMOT,
  META_NONMOT,
  META_PEDESTRIAN,
  MAX_META_TYPE
};

enum class ObjectType {
    UNKNOWN = 0,
    UNKNOWN_MOVABLE = 1,
    UNKNOWN_UNMOVABLE = 2,
    PEDESTRIAN = 3,
    BICYCLE = 4,
    VEHICLE = 5,
    MAX_OBJECT_TYPE = 6,
};


//struct Obstacle {
//  std::vector<int> grids;
//  apollo::perception::pcl_util::PointCloudPtr cloud;
//  float score;
//  float height;
//  MetaType meta_type;
//  std::vector<float> meta_type_probs;
//
//  Obstacle() : score(0.0), height(-5.0), meta_type(MetaType::META_UNKNOWN) {
//    cloud.reset(new apollo::perception::pcl_util::PointCloud);
//    meta_type_probs.assign(static_cast<int>(MetaType::MAX_META_TYPE), 0.0);
//  }
//};

class Cluster2D {
 public:
  Cluster2D() = default;
  ~Cluster2D() = default;


    bool Init(int rows, int cols, float range) {
    rows_ = rows;
    cols_ = cols;
    grids_ = rows_ * cols_;
    range_ = range;
    scale_ = 0.5 * static_cast<float>(rows_) / range_;
    inv_res_x_ = 0.5 * static_cast<float>(cols_) / range_;
    inv_res_y_ = 0.5 * static_cast<float>(rows_) / range_;
    points_num = 0;
    point2grid_.clear();
    obstacles_.clear();
    grids.clear();
    id_img_.assign(grids_, -1);
    score_.clear();
    height_.clear();
    cloud_.clear();
    return_cloud.clear();
    return_score.clear();
    type_name.clear();
    valid_indices_in_pc_.clear();
    pc_ptr_.clear();
    return true;
  }

  void Cluster(const float* category_pt_data,
               const float* instance_pt_x_data,
               const float* instance_pt_y_data,
               const float* pc_ptr,
               const int* valid_indices,
               const int cloud_size,
               float objectness_thresh, bool use_all_grids_for_clustering) {
    valid_indices_in_pc_.clear();
    pc_ptr_.clear();
    for(int c=0; c < cloud_size; ++c) pc_ptr_.push_back(pc_ptr[c]);
    std::vector<std::vector<Node>> nodes(rows_,
                                         std::vector<Node>(cols_, Node()));
    // map points into grids
    int tot_point_num = cloud_size / 4;
    for(int j=0; j < tot_point_num; ++j) valid_indices_in_pc_.push_back(valid_indices[j]);
    point2grid_.assign(static_cast<int>(tot_point_num), -1);

    for (int i = 0; i < tot_point_num; ++i) {
      int point_id = valid_indices_in_pc_[i];
      CHECK_GE(point_id, 0);
      CHECK_LT(point_id, static_cast<int>(tot_point_num));
      // * the coordinates of x and y have been exchanged in feature generation
      // step,
      // so we swap them back here.
      int pos_x = F2I(pc_ptr_[point_id*4 + 1], range_, inv_res_x_);  // col
      int pos_y = F2I(pc_ptr_[point_id*4], range_, inv_res_y_);  // row
      if (IsValidRowCol(pos_y, pos_x)) {
        // get grid index and count point number for corresponding node
        point2grid_[i] = RowCol2Grid(pos_y, pos_x);
        nodes[pos_y][pos_x].point_num++;
      }
    }

    // construct graph with center offset prediction and objectness
    for (int row = 0; row < rows_; ++row) {
      for (int col = 0; col < cols_; ++col) {
        int grid = RowCol2Grid(row, col);
        Node* node = &nodes[row][col];
        DisjointSetMakeSet(node);
        node->is_object =
            (use_all_grids_for_clustering || nodes[row][col].point_num > 0) &&
            (*(category_pt_data + grid) >= objectness_thresh);
        int center_row = std::round(row + instance_pt_x_data[grid] * scale_);
        int center_col = std::round(col + instance_pt_y_data[grid] * scale_);
        center_row = std::min(std::max(center_row, 0), rows_ - 1);
        center_col = std::min(std::max(center_col, 0), cols_ - 1);
        node->center_node = &nodes[center_row][center_col];
      }
    }

    // traverse nodes
    for (int row = 0; row < rows_; ++row) {
      for (int col = 0; col < cols_; ++col) {
        Node* node = &nodes[row][col];
        if (node->is_object && node->traversed == 0) {
          Traverse(node);
        }
      }
    }
    for (int row = 0; row < rows_; ++row) {
      for (int col = 0; col < cols_; ++col) {
        Node* node = &nodes[row][col];
        if (!node->is_center) {
          continue;
        }
        for (int row2 = row - 1; row2 <= row + 1; ++row2) {
          for (int col2 = col - 1; col2 <= col + 1; ++col2) {
            if ((row2 == row || col2 == col) && IsValidRowCol(row2, col2)) {
              Node* node2 = &nodes[row2][col2];
              if (node2->is_center) {
                DisjointSetUnion(node, node2);
              }
            }
          }
        }
      }
    }

    int count_obstacles = 0;
    obstacles_.clear();
    id_img_.assign(grids_, -1);
    for (int row = 0; row < rows_; ++row) {
      for (int col = 0; col < cols_; ++col) {
        Node* node = &nodes[row][col];
        if (!node->is_object) {
          continue;
        }
        Node* root = DisjointSetFind(node);
        if (root->obstacle_id < 0) {
          root->obstacle_id = count_obstacles++;
          CHECK_EQ(static_cast<int>(obstacles_.size()), count_obstacles - 1);
          std::vector<int> temp;
          obstacles_.push_back(temp);
        }
        int grid = RowCol2Grid(row, col);
        CHECK_GE(root->obstacle_id, 0);
        id_img_[grid] = root->obstacle_id;
        obstacles_[root->obstacle_id].push_back(grid);
      }
    }
    CHECK_EQ(static_cast<size_t>(count_obstacles), obstacles_.size());
  }



  void Filter(const float* confidence_pt_data,
              const float* height_pt_data) {
    for (size_t obstacle_id = 0; obstacle_id < obstacles_.size();
         obstacle_id++) {
      double score = 0.0;
      double height = 0.0;
      for (int grid : obstacles_[obstacle_id]) {
        score += static_cast<double>(confidence_pt_data[grid]);
        height += static_cast<double>(height_pt_data[grid]);
      }
      score_.push_back(score / static_cast<double>(obstacles_[obstacle_id].size()));
      height_.push_back(height / static_cast<double>(obstacles_[obstacle_id].size()));
    }
  }

  void Classify(const float* classify_pt_data) {
    int num_classes = 5;
    meta_type.clear();
    meta_type_probs.clear();
    meta_type_probs.resize(static_cast<int>(obstacles_.size()));
    for (int i=0; i < static_cast<int>(obstacles_.size()); ++i) meta_type_probs[i].assign(num_classes, 0);
    CHECK_EQ(num_classes, static_cast<int>(MetaType::MAX_META_TYPE));
    for (size_t obs_id = 0; obs_id < obstacles_.size(); obs_id++) {
      for (size_t grid_id = 0; grid_id < obstacles_[obs_id].size(); grid_id++) {
        int grid = obstacles_[obs_id][grid_id];
        for (int k = 0; k < num_classes; k++) {
          meta_type_probs[obs_id][k] += classify_pt_data[k * grids_ + grid];
        }
      }
      int meta_type_id = 0;
      for (int k = 0; k < num_classes; k++) {

        meta_type_probs[obs_id][k] /= obstacles_[obs_id].size();
        if (meta_type_probs[obs_id][k] > meta_type_probs[obs_id][meta_type_id]) {
          meta_type_id = k;
        }
      }
      meta_type.push_back(static_cast<MetaType>(meta_type_id));
    }
  }

  void GetObjects(const float confidence_thresh, const float height_thresh,
                  const int min_pts_num) {
    cloud_.clear();
    cloud_.resize(static_cast<int>(obstacles_.size()));
    return_score.clear();
    return_cloud.clear();
    type_name.clear();
    for (size_t i = 0; i < point2grid_.size(); ++i) {
      int grid = point2grid_[i];
      if (grid < 0) {
        continue;
      }
      CHECK_GE(grid, 0);
      CHECK_LT(grid, grids_);
      int obstacle_id = id_img_[grid];

      int point_id = valid_indices_in_pc_[i];
      CHECK_GE(point_id, 0);
      if (obstacle_id >= 0 &&
          score_[obstacle_id] >= confidence_thresh) {
        if (height_thresh < 0 ||
            pc_ptr_[point_id*4 + 2] <=
                height_[obstacle_id] + height_thresh) {
            //std::cout << "id: " << point_id << " " << "z: " << pc_ptr_[point_id*4 +2] << std::endl;
          cloud_[obstacle_id].push_back(pc_ptr_[point_id*4]);
          cloud_[obstacle_id].push_back(pc_ptr_[point_id*4 + 1]);
          cloud_[obstacle_id].push_back(pc_ptr_[point_id*4 + 2]);
          cloud_[obstacle_id].push_back(pc_ptr_[point_id*4 + 3]);
        }
      }
    }
    for (size_t obstacle_id = 0; obstacle_id < obstacles_.size();
         obstacle_id++) {
      int obj_size = cloud_[obstacle_id].size() / 4;
      if (obj_size < min_pts_num) {
        continue;
      }
//      std::cout << " for debug" <<std::endl;
      return_score.push_back(score_[obstacle_id]);
      return_cloud.push_back(cloud_[obstacle_id]);
      MetaType id_tmp = meta_type[obstacle_id];
      ObjectType tmp = GetObjectType(id_tmp);
      std::string tmp_name = GetTypeText(tmp);
      type_name.push_back(tmp_name);
    }
  }

 private:

  struct Node {
    Node* center_node;
    Node* parent;
    char node_rank;
    char traversed;
    bool is_center;
    bool is_object;
    int point_num;
    int obstacle_id;


    Node() {
      center_node = nullptr;
      parent = nullptr;
      node_rank = 0;
      traversed = 0;
      is_center = false;
      is_object = false;
      point_num = 0;
      obstacle_id = -1;
    }
  };

  inline bool IsValidRowCol(int row, int col) const {
    return IsValidRow(row) && IsValidCol(col);
  }

  inline bool IsValidRow(int row) const { return row >= 0 && row < rows_; }

  inline bool IsValidCol(int col) const { return col >= 0 && col < cols_; }

  inline int RowCol2Grid(int row, int col) const { return row * cols_ + col; }

  void Traverse(Node* x) {
    std::vector<Node*> p;
    p.clear();
    while (x->traversed == 0) {
      p.push_back(x);
      x->traversed = 2;
      x = x->center_node;
    }
    if (x->traversed == 2) {
      for (int i = static_cast<int>(p.size()) - 1; i >= 0 && p[i] != x; i--) {
        p[i]->is_center = true;
      }
      x->is_center = true;
    }
    for (size_t i = 0; i < p.size(); i++) {
      Node* y = p[i];
      y->traversed = 1;
      y->parent = x->parent;
    }
  }

  ObjectType GetObjectType(const MetaType meta_type_id) {
    switch (meta_type_id) {
      case MetaType::META_UNKNOWN:
        return ObjectType::UNKNOWN;
      case MetaType::META_SMALLMOT:
        return ObjectType::VEHICLE;
      case MetaType::META_BIGMOT:
        return ObjectType::VEHICLE;
      case MetaType::META_NONMOT:
        return ObjectType::BICYCLE;
      case MetaType::META_PEDESTRIAN:
        return ObjectType::PEDESTRIAN;
      default: {
        AERROR << "Undefined ObjectType output by CNNSeg model.";
        return ObjectType::UNKNOWN;
      }
    }
  }
  //add extra moduel
  std::string GetTypeText(ObjectType type) {
  if (type == ObjectType::VEHICLE) {
    return "car";
  }
  if (type == ObjectType::PEDESTRIAN) {
    return "pedestrian";
  }
  if (type == ObjectType::BICYCLE) {
    return "bicycle";
  }

  return "unknown";
  }

  std::vector<float> GetObjectTypeProbs(
      const std::vector<float>& meta_type_probs) {
    std::vector<float> object_type_probs(
        static_cast<int>(ObjectType::MAX_OBJECT_TYPE), 0.0);
    object_type_probs[static_cast<int>(ObjectType::UNKNOWN)] =
        meta_type_probs[static_cast<int>(MetaType::META_UNKNOWN)];
    object_type_probs[static_cast<int>(ObjectType::VEHICLE)] =
        meta_type_probs[static_cast<int>(MetaType::META_SMALLMOT)] +
        meta_type_probs[static_cast<int>(MetaType::META_BIGMOT)];
    object_type_probs[static_cast<int>(ObjectType::BICYCLE)] =
        meta_type_probs[static_cast<int>(MetaType::META_NONMOT)];
    object_type_probs[static_cast<int>(ObjectType::PEDESTRIAN)] =
        meta_type_probs[static_cast<int>(MetaType::META_PEDESTRIAN)];
    return object_type_probs;
  }

  int rows_;
  int cols_;
  int grids_;
  float range_;
  float scale_;
  float inv_res_x_;
  float inv_res_y_;
  int points_num;

  std::vector<float> pc_ptr_;

  std::vector<int> point2grid_;
  std::vector<int> id_img_;
public:
  std::vector<std::vector<int>> obstacles_;
  std::vector<int> grids;
  std::vector<std::vector<float>> meta_type_probs;
  std::vector<MetaType> meta_type;
  std::vector<int> valid_indices_in_pc_;
  std::vector<float> score_;
  std::vector<float> height_;
  std::vector<std::vector<float>> cloud_;

  //for return
  std::vector<std::vector<float>> return_cloud;
  std::vector<float> return_score;
  std::vector<std::string> type_name;

};

}  // namespace apollo

#endif  // MODULES_PERCEPTION_OBSTACLE_LIDAR_SEGMENTATION_CNNSEG_CLUSTER2D_H_
