//
// Created by bitfrog on 10/28/19.
//

#pragma once
#include "resnet_matrix.h"
#include "resnet_config.h"
#include <cmath>
#include <limits>

namespace Resnet {
class Math {
 public:
  static Matrix3 Conv(const Matrix3 &in_data, const Matrix3 &kernel, Convolution conv_para);
  static Matrix3 PadMatrix(const Matrix3 &in_data, int pad);
  static float Feature(const Matrix3& in_data, const Matrix3& kernel, int channel_no, \
                      int in_channel_no, int row, int col, int stride);
  static std::vector<Matrix3> BN(const std::vector<Matrix3>& in_data, float maf, float eps, bool scale_bias);
  static Matrix3 Pooling(const Matrix3& in_data, int kernel_size, int stride, std::string type);
  static float PoolingFeature(const Matrix3& in_data, int kernel_size, int stride,
                                std::string type, int ch, int row, int col);
  static Matrix3 ReLU(const Matrix3& in_data, float value);
  static Matrix3 FullConnect(const Matrix3& in_data, const InnerProduct& inner);
  static Matrix3 Reshape(const Matrix3& in_data, int ch, int row, int col);
  static Matrix3 Softmax(const Matrix3& in_data);
  static Matrix3 CalcExp(const Matrix3& in_data);
};
} // namespace Resnet
