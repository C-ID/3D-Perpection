
#ifndef MODULES_PERCEPTION_OBSTACLE_BASE_OBJECT_H_
#define MODULES_PERCEPTION_OBSTACLE_BASE_OBJECT_H_

#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"

#include "pcl_types.h"
#include "types.h"


namespace apollo {
namespace perception {

struct alignas(16) Object {
  Object(){
  cloud.reset(new pcl_util::PointCloud);
  type_probs.resize(static_cast<int>(ObjectType::MAX_OBJECT_TYPE), 0);
  };

  // object id per frame
  int id = 0;
  // point cloud of the object
  pcl_util::PointCloudPtr cloud;
  // convex hull of the object
  PolygonDType polygon;

  // oriented boundingbox information
  // main direction
  Eigen::Vector3d direction = Eigen::Vector3d(1, 0, 0);
  // the yaw angle, theta = 0.0 <=> direction = (1, 0, 0)
  double theta = 0.0;
  // ground center of the object (cx, cy, z_min)
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  // size of the oriented bbox, length is the size in the main direction
  double length = 0.0;
  double width = 0.0;
  double height = 0.0;
  // shape feature used for tracking
  std::vector<float> shape_features;

  // foreground score/probability
  float score = 0.0;
  // foreground score/probability type
  ScoreType score_type = ScoreType::SCORE_CNN;

  // Object classification type.
  ObjectType type = ObjectType::UNKNOWN;
  // Probability of each type, used for track type.
  std::vector<float> type_probs;

  // fg/bg flag
  bool is_background = false;

  //type name
  std::string type_name;

  // tracking information
  int track_id = 0;
  //Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  // age of the tracked object
  double tracking_time = 0.0;
  double latest_tracked_time = 0.0;

  // CIPV
  bool b_cipv = false;
};

typedef std::shared_ptr<Object> ObjectPtr;
typedef std::shared_ptr<const Object> ObjectConstPtr;

}  // namespace perception
}  // namespace apollo

#endif  // MODULES_PERCEPTION_OBSTACLE_BASE_OBJECT_H_
