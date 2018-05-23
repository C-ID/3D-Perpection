

#ifndef FEATURE_GENERATOR_H_  // NOLINT
#define FEATURE_GENERATOR_H_  // NOLINT

#include <cmath>
#include <string>
#include <vector>
//#include <jsoncpp/json/json.h>


namespace apollo {
namespace perception {
namespace cnnseg {

class FeatureGenerator {
 public:
  FeatureGenerator() {}

  ~FeatureGenerator() {}

  bool Init();
  void Generate(float* points,const int length, float* feature);

  inline float Pixel2Pc(int in_pixel, float in_size, float out_range)
  {
      float res = 2.0 * out_range / in_size;
      return out_range - (static_cast<float>(in_pixel) + 0.5f) * res;
  }

  inline int F2I(float val, float ori, float scale)
  {
      return static_cast<int>(std::floor((ori - val) * scale));
  }


private:
  float LogCount(int count)
  {
    if (count < static_cast<int>(log_table_.size())) return log_table_[count];
    return std::log(static_cast<float>(1 + count));
  }

  std::vector<float> log_table_;

  int width_ = 0;
  int height_ = 0;
  int range_ = 0;
  int channels = 0;

  float min_height_ = 0.0;
  float max_height_ = 0.0;


  // point index in feature map
  std::vector<int> map_idx_;

  //eight channels feature
  std::vector<float> max_height_data_;
  std::vector<float> mean_height_data_;
  std::vector<float> count_data_;
  std::vector<float> direction_data_;
  std::vector<float> top_intensity_data_;
  std::vector<float> mean_intensity_data_;
  std::vector<float> distance_data_;
  std::vector<float> nonempty_data_;

//  Json::Value root;
//  Json::Value pts;
//  Json::StyledWriter writer;
};

}  // namespace cnnseg
}  // namespace perception
}  // namespace apollo

#endif
