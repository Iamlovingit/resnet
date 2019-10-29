//
// Created by bitfrog on 10/28/19.
//

#pragma once
#include "resnet_matrix.h"
#include "resnet_config.h"

namespace Resnet {
class Math {
 public:
  static Matrix3 Conv(const Matrix3 &in_data, const Matrix3 &kernel, Convolution conv_para);
  static Matrix3 PadMatrix(const Matrix3 &in_data, int pad);
  static float Feature(const Matrix3& in_data, const Matrix3& kernel, int channel_no, \
                      int in_channel_no, int row, int col, int stride);
};
} // namespace Resnet
