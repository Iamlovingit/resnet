//
// Created by bitfrog on 10/28/19.
//

#include <string>
#include "resnet_matrix.h"
#include "stdlib.h"
#include "time.h"
namespace Resnet {
#define MAX_RAND 99
Matrix3::Matrix3(int row_size, int col_size, int channel_size) {
  channel_size_ = channel_size;
  row_size_ = row_size;
  col_size_ = col_size;

  srand(time(NULL));
  mat_ = (float***) malloc(channel_size * sizeof(float**));
  for(int i = 0; i < channel_size; i++) {
    mat_[i] = (float**) malloc(row_size * sizeof(float*));
    for(int j = 0; j < row_size; j++) {
      mat_[i][j] = (float*) malloc(col_size * sizeof(float));
      for(int k = 0; k < col_size; k++) {
        mat_[i][j][k] = rand()%(MAX_RAND+1);
      }
    }
  }
}

Matrix3::Matrix3() {
  channel_size_ = 0;
  row_size_ = 0;
  col_size_ = 0;
  mat_ = nullptr;
}

void Matrix3::operator=(Resnet::Matrix3 matrix) {
  this->channel_size_ = matrix.channel_size_;
  this->row_size_ = matrix.row_size_;
  this->col_size_ = matrix.col_size_;
  this->mat_ = matrix.mat_;
}
} //namespace Resnet