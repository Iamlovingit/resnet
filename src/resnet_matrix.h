//
// Created by bitfrog on 10/28/19.
//

#pragma once
#include <vector>
#include "resnet_hpc_matrix_multi.h"
namespace Resnet {

class Matrix2 {
 public:
  Matrix2(int,int);
  ~Matrix2();
 public:
  int row_size_;
  int col_size_;
  float ** mat_;
};


class Matrix3 {
 public:
  Matrix3(int,int,int);
  Matrix3(int,int,int,float);
  Matrix3();
  ~Matrix3();
  void operator= (const Matrix3&);
  Matrix3 operator+ (const Matrix3& m1);
  Matrix3 operator* (const Matrix3& m1);
  void operator-= (const double& value);
  Matrix3 operator/(const float& value);
 public:
  int channel_size_;
  int row_size_;
  int col_size_;
  int size_;
  std::vector<float> mat_;
};
} // namespace Resnet
