
#ifndef UTIL_H_
#define UTIL_H_

#include <string>

namespace apollo {
namespace perception {
namespace cnnseg {

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

}  // namespace cnnseg
}  // namespace perception
}  // namespace apollo

#endif  // MODULES_PERCEPTION_OBSTACLE_LIDAR_SEGMENTATION_CNNSEG_UTIL_H_
