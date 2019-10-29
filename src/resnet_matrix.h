//
// Created by bitfrog on 10/28/19.
//

#pragma once

namespace Resnet {
class Matrix3 {
 public:
  Matrix3(int,int,int);
  Matrix3();
  void operator= (Matrix3);
 public:
  int channel_size_;
  int row_size_;
  int col_size_;
  float*** mat_;
};
} // namespace Resnet
