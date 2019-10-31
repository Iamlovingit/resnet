//
// Created by bitfrog on 10/28/19.
//

#include <string>
#include "resnet_matrix.h"
#include <cstdlib>
#include <ctime>
namespace Resnet {
#define MAX_RAND 99
Matrix3::Matrix3(int channel_size, int row_size, int col_size) {
  channel_size_ = channel_size;
  row_size_ = row_size;
  col_size_ = col_size;

  srand(time(nullptr));
  mat_ = (float***) malloc(channel_size * sizeof(float**));
  for(int i = 0; i < channel_size; i++) {
    mat_[i] = (float**) malloc(row_size * sizeof(float*));
    for(int j = 0; j < row_size; j++) {
      mat_[i][j] = (float*) malloc(col_size * sizeof(float));
      for(int k = 0; k < col_size; k++) {
        mat_[i][j][k] = rand()%(MAX_RAND+1)/(float)(MAX_RAND+1) - 0.5; //-0.5 ~ 0.5
      }
    }
  }
}

Matrix3::Matrix3(int channel_size, int row_size, int col_size, float value) {
  channel_size_ = channel_size;
  row_size_ = row_size;
  col_size_ = col_size;

  mat_ = (float***) malloc(channel_size * sizeof(float**));
  for(int i = 0; i < channel_size; i++) {
    mat_[i] = (float**) malloc(row_size * sizeof(float*));
    for(int j = 0; j < row_size; j++) {
      mat_[i][j] = (float*) malloc(col_size * sizeof(float));
      for(int k = 0; k < col_size; k++) {
        mat_[i][j][k] = value;
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

void Matrix3::operator=(const Resnet::Matrix3 &matrix) {
  this->channel_size_ = matrix.channel_size_;
  this->row_size_ = matrix.row_size_;
  this->col_size_ = matrix.col_size_;

  this->mat_ = (float***) malloc(channel_size_ * sizeof(float**));
  for(int i = 0; i < channel_size_; i++) {
    this->mat_[i] = (float**) malloc(row_size_ * sizeof(float*));
    for(int j = 0; j < row_size_; j++) {
      this->mat_[i][j] = (float*) malloc(col_size_ * sizeof(float));
      for(int k = 0; k < col_size_; k++) {
        this->mat_[i][j][k] = matrix.mat_[i][j][k];
      }
    }
  }
}

Matrix3 Matrix3::operator+(const Resnet::Matrix3 &matrix) {
  Matrix3 out(this->channel_size_, this->row_size_, this->col_size_);
  for(int i = 0; i < channel_size_; i++) {
    for(int j = 0; j < row_size_; j++) {
      for(int k = 0; k < col_size_; k++) {
        out.mat_[i][j][k] = this->mat_[i][j][k]+matrix.mat_[i][j][k];
      }
    }
  }
  return out;
}


Matrix3 Matrix3::operator-= (const double &value) {
  Matrix3 out(this->channel_size_, this->row_size_, this->col_size_);
  for(int i = 0; i < channel_size_; i++) {
    for(int j = 0; j < row_size_; j++) {
      for(int k = 0; k < col_size_; k++) {
        out.mat_[i][j][k] = this->mat_[i][j][k]-value;
      }
    }
  }
  return out;
}

Matrix3 Matrix3::operator*(const Resnet::Matrix3 &matrix) {
  Matrix3 out(this->channel_size_, this->row_size_, matrix.col_size_, 0);
  for(int i = 0; i < out.channel_size_; i++) {
    for(int j = 0; j < out.row_size_; j++) {
      for(int k = 0; k < out.col_size_; k++) {
        for(int n = 0; n < matrix.row_size_; n++)
          out.mat_[i][j][k] += this->mat_[i][j][n]*matrix.mat_[i][n][k];
      }
    }
  }
  return out;
}

Matrix3 Matrix3::operator/(const float &value) {
  Matrix3 out(this->channel_size_, this->row_size_, this->col_size_, 0);
  for(int i = 0; i < out.channel_size_; i++) {
    for(int j = 0; j < out.row_size_; j++) {
      for(int k = 0; k < out.col_size_; k++) {
          out.mat_[i][j][k] = this->mat_[i][j][k] / value;
      }
    }
  }
  return out;
}

Matrix3::~Matrix3() {
//  for(int i = 0; i < channel_size_; i++) {
//    for(int j = 0; j < row_size_; j++) {
//      if(mat_[i][j] != nullptr) {
//        free(mat_[i][j]);
//      }
//    }
//  }
}


/*********************************************************/
/*                       Matrix2                         */
/*********************************************************/
Matrix2::Matrix2(int row_size, int col_size) {
  row_size_ = row_size;
  col_size_ = col_size;

  mat_ = (float**)malloc(row_size_ * sizeof(float*));
  for(int i = 0 ; i < row_size_; i++) {
    mat_[i] = (float*)malloc(col_size_ * sizeof(float));
    for(int j = 0; j < col_size_; j++) {
      mat_[i][j] = 0;
    }
  }
}


} //namespace Resnet