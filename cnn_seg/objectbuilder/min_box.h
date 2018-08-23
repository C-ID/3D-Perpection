

#ifndef MIN_BOX_H
#define MIN_BOX_H

#include <string>
#include <vector>

#include "object.h"
#include "base_object_builder.h"

namespace apollo {
namespace perception {

class MinBoxObjectBuilder : public BaseObjectBuilder {
 public:
  MinBoxObjectBuilder() : BaseObjectBuilder() {}
  virtual ~MinBoxObjectBuilder() {}

  bool Init() override {
    return true;
  }

  bool Build(const ObjectBuilderOptions& options,
             std::vector<ObjectPtr>* objects) override;
  std::string name() const override {
    return "MinBoxObjectBuilder";
  }

 protected:
  void BuildObject(ObjectBuilderOptions options, ObjectPtr object);

  void ComputePolygon2dxy(ObjectPtr obj);

  double ComputeAreaAlongOneEdge(ObjectPtr obj, size_t first_in_point,
                                 Eigen::Vector3d* center, double* lenth,
                                 double* width, Eigen::Vector3d* dir);

  void ReconstructPolygon(const Eigen::Vector3d& ref_ct, ObjectPtr obj);

  void ComputeGeometricFeature(const Eigen::Vector3d& ref_ct, ObjectPtr obj);

 private:
  DISALLOW_COPY_AND_ASSIGN(MinBoxObjectBuilder);
};

// Register plugin.
//REGISTER_OBJECTBUILDER(MinBoxObjectBuilder);

}  // namespace perception
}  // namespace apollo

#endif  // MODULES_PERCEPTION_OBSTACLE_LIDAR_OBJECT_BUILDER_MIN_BOX_H
