//
// Created by bitfrog on 10/28/19.
//

#include <string>
#include <iostream>
#include "resnet_matrix.h"
#include <cstdlib>
#include <ctime>
namespace Resnet {
#define MAX_RAND 99
Matrix3::Matrix3(int channel_size, int row_size, int col_size) {
  channel_size_ = channel_size;
  row_size_ = row_size;
  col_size_ = col_size;
  size_ = channel_size_*row_size_*col_size_;
  std::vector<float>().swap(mat_);
  srand(time(nullptr));
  for(int i = 0; i < size_; i++) {
    mat_.push_back(rand()%(MAX_RAND+1)/(float)(MAX_RAND+1) - 0.5); //-0.5 ~ 0.5
  }
}

Matrix3::Matrix3(int channel_size, int row_size, int col_size, float value) {
  channel_size_ = channel_size;
  row_size_ = row_size;
  col_size_ = col_size;
  size_ = channel_size_*row_size_*col_size_;
  std::vector<float>().swap(mat_);
  for(int i = 0; i < size_; i++) {
    mat_.push_back(value); //-0.5 ~ 0.5
  }
}

Matrix3::Matrix3() {
  channel_size_ = 0;
  row_size_ = 0;
  col_size_ = 0;
  size_ = 0;
  std::vector<float>().swap(mat_);;
}

void Matrix3::operator=(const Resnet::Matrix3 &matrix) {
  this->channel_size_ = matrix.channel_size_;
  this->row_size_ = matrix.row_size_;
  this->col_size_ = matrix.col_size_;
  this->size_ = matrix.size_;
  std::vector<float>().swap(mat_);;
  for(int i = 0; i < size_ ; i++) {
    mat_.push_back(matrix.mat_[i]);
  }
}

Matrix3 Matrix3::operator+(const Resnet::Matrix3 &matrix) {
  Matrix3 out(this->channel_size_, this->row_size_, this->col_size_,0);
  for(int i = 0; i < out.size_; i++) {
    out.mat_[i] = this->mat_[i]+matrix.mat_[i];
  }
  return out;
}


void Matrix3::operator-= (const double &value) {
  for(int i = 0; i < size_; i++) {
    mat_[i] -= value;
  }
}

Matrix3 Matrix3::operator*(const Resnet::Matrix3 &matrix) {
  Matrix3 out(this->channel_size_, this->row_size_, matrix.col_size_, 0);
  for(int i = 0; i < out.channel_size_; i++) {
    for(int j = 0; j < out.row_size_; j++) {
      for(int k = 0; k < out.col_size_; k++) {
        for(int n = 0; n < matrix.row_size_; n++)
          out.mat_[i*row_size_*col_size_+j*col_size_+k] =
            this->mat_[i*this->row_size_*this->col_size_+j*this->col_size_+n] *
            matrix.mat_[i*matrix.row_size_*matrix.col_size_+n*matrix.col_size_+k];
      }
    }
  }
  return out;
}

Matrix3 Matrix3::operator/(const float &value) {
  Matrix3 out(this->channel_size_, this->row_size_, this->col_size_, 0);
  for(int i = 0; i < out.size_; i++) {
    out.mat_[i] = this->mat_[i]/value;
  }
  return out;
}

Matrix3::~Matrix3(){
  size_ = 0;
  channel_size_ = 0;
  row_size_ = 0;
  col_size_ = 0;
  std::vector<float>().swap(mat_);;
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

Matrix2::~Matrix2() {
  for(int i = 0; i < row_size_; i++) {
    free(mat_[i]);
  }
  free(mat_);
  mat_ = nullptr;
  col_size_ = 0;
  row_size_ = 0;
}

} //namespace Resnet