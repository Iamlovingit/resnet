//
// Created by bitfrog on 10/28/19.
//

#pragma once

namespace Resnet {

class Matrix2 {
 public:
  Matrix2(int,int);
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
 public:
  int channel_size_;
  int row_size_;
  int col_size_;
  float*** mat_;
};
} // namespace Resnet
