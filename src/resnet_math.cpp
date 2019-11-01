//
// Created by bitfrog on 10/28/19.
//

#include "resnet_math.h"
#include <iostream>

namespace  Resnet {
Matrix3 Math::Conv(const Matrix3 &in_data, const Matrix3 &kernel, Convolution conv_para) {
  std::cout << "Conv debug : indata.row=" << in_data.row_size_ << ", kernel.size=" << conv_para.kernel_size << ", stride=" << conv_para.stride << ", pad = " << conv_para.pad <<  std::endl;
  Matrix3 pad_m = PadMatrix(in_data, conv_para.pad); //padding input matrix
  int out_size = (in_data.row_size_ + 2*conv_para.pad - conv_para.kernel_size)/conv_para.stride+1;
  Matrix3 out(conv_para.output_size, out_size, out_size, 0); // init out matrix
  //value every element.
  for(int ch = 0; ch < out.channel_size_; ch++) {
    for(int c = 0; c < in_data.channel_size_; c++) { // every input channel
       for(int row = 0; row < out.row_size_; row++) { //every row
         for(int col = 0; col < out.col_size_; col++) { //every col
           //out.mat_[ch][row][col] += Feature(pad_m, kernel, ch, c, row, col, conv_para.stride);
           int idx = ch*out.row_size_*out.col_size_ + row*out.col_size_ + col;
           out.mat_[idx] += Feature(pad_m, kernel, ch, c, row, col, conv_para.stride);
         }
       }
    }
  }
  return out;
}

Matrix3 Math::PadMatrix(const Resnet::Matrix3 &in_data, int pad) {
  Matrix3 out(in_data.channel_size_, in_data.row_size_+2*pad, in_data.col_size_+2*pad);
  for(int ch = 0; ch < out.channel_size_; ch++) {
    for(int row = 0; row < out.row_size_; row++) {
      for(int col = 0; col < out.col_size_; col++) {
        int out_idx = ch*out.row_size_*out.col_size_ + row*out.col_size_ + col;
        int in_idx = ch*in_data.row_size_*in_data.col_size_+(row-pad)*in_data.col_size_+(col-pad);
        if(row < pad || col < pad || row > out.row_size_-pad-1 || col > out.col_size_-pad-1) {
          out.mat_[out_idx] = 0;
        } else {
          out.mat_[out_idx] = in_data.mat_[in_idx];
        }
      }
    }
  }
  return out;
}

float Math::Feature(const Resnet::Matrix3 &data, const Resnet::Matrix3 &kernel, int channel_no, int in_channel_no,
                    int row, int col, int stride) {
  float value=0;
  Matrix2 data_src(kernel.row_size_,kernel.col_size_);
  Matrix2 ker_src(kernel.row_size_,kernel.col_size_);

  //init data_src
  for(int i = 0; i < data_src.row_size_; i++) {
    for(int j = 0; j < data_src.col_size_; j++) {
      int idx = in_channel_no*data.row_size_*data.col_size_ + (row*stride+1)*data.col_size_+(col*stride+j);
      data_src.mat_[i][j] = data.mat_[idx];
    }
  }

  //init ker_src
  for(int i = 0; i < data_src.row_size_; i++) {
    for(int j = 0; j < data_src.col_size_; j++) {
      int idx = (channel_no*data.channel_size_+in_channel_no)*kernel.row_size_*kernel.col_size_ + i*kernel.col_size_ +j;
      ker_src.mat_[i][j] = kernel.mat_[idx];
    }
  }

  //calc value
  for(int i = 0; i < data_src.row_size_; i++) {
    for(int j = 0; j < data_src.col_size_; j++)
      value += data_src.mat_[i][j]*ker_src.mat_[i][j];
  }
  return value;
}

std::vector<Matrix3> Math::BN(const std::vector<Resnet::Matrix3> &in_data, float maf, float eps, bool scale_bias) {
  std::vector<Matrix3> out;
  Matrix3 mean(in_data[0].channel_size_, in_data[0].row_size_, in_data[0].col_size_, 0);
  std::cout << "infunc bn, in_data size = " << in_data.size() << std::endl;
  for(auto it : in_data) {
    std::cout << "sss in_data.size3=" << it.channel_size_ << "," <<  it.row_size_ << "," << it.col_size_ << std::endl;
     mean = mean + it;
  }
  mean = mean/in_data.size();

  Matrix3 sqrt_mat(mean.channel_size_, mean.row_size_, mean.col_size_,0);
  for(int ch = 0; ch < in_data[0].channel_size_; ch++) {
    for(int row = 0; row < in_data[0].row_size_; row++) {
      for(int col = 0; col < in_data[0].col_size_; col++) {
        for(int i = 0; i < in_data.size(); i++) {
          int s_idx = ch*sqrt_mat.row_size_*sqrt_mat.col_size_ + row*sqrt_mat.col_size_ + col;
          int i_idx = ch*in_data[i].row_size_*in_data[i].col_size_ + row*in_data[i].col_size_ + col;
          int m_idx = ch*mean.row_size_*mean.col_size_ + row*mean.col_size_ + col;
          sqrt_mat.mat_[s_idx] += pow(in_data[i].mat_[i_idx]-mean.mat_[m_idx], 2);
        }
        int s_idx = ch*sqrt_mat.row_size_*sqrt_mat.col_size_ + row*sqrt_mat.col_size_ + col;
        sqrt_mat.mat_[s_idx] /= in_data.size();
        sqrt_mat.mat_[s_idx] = (float)sqrt(sqrt_mat.mat_[s_idx]+1e-4);
      }
    }
  }

  for(int i = 0; i < in_data.size(); i++) {
    Matrix3 mat(mean.channel_size_, mean.row_size_, mean.col_size_, 0);
    for(int ch = 0; ch < in_data[0].channel_size_; ch++) {
      for (int row = 0; row < in_data[0].row_size_; row++) {
        for (int col = 0; col < in_data[0].col_size_; col++) {
          int mat_idx = ch*mat.row_size_*mat.col_size_ + row*mat.col_size_ + col;
          int in_idx = ch*in_data[i].row_size_*in_data[i].col_size_ + row*in_data[i].col_size_+ col;
          int mean_idx = ch*mean.row_size_*mean.col_size_ + row*mean.col_size_ + col;
          int s_idx = ch*sqrt_mat.row_size_*sqrt_mat.col_size_ + row*sqrt_mat.col_size_ + col;
          mat.mat_[mat_idx] = 1e-2 * (in_data[i].mat_[in_idx] - mean.mat_[mean_idx]) \
                                   /sqrt_mat.mat_[s_idx] + 1e-2;
        }
      }
    }
    out.push_back(mat);
  }
  return out;
}

Matrix3 Math::Pooling(const Resnet::Matrix3 &in_data, int kernel_size, int stride, std::string type) {
  //calc out row,col
  std::cout << "Pooling debug, in size=" << in_data.row_size_ << ", ker size = " << kernel_size << "stride ="<< stride <<std::endl;
  int row = (in_data.row_size_ - kernel_size)/stride + 1;
//  if(row < 1) row = 1;
  int col = row;
  Matrix3 out(in_data.channel_size_, row, col, 0);
  for(int ch = 0; ch < out.channel_size_; ch++) {
    for(int r = 0; r < row; r++) {
      for(int c = 0; c < col; c++) {
        int idx = ch*out.row_size_*out.col_size_ + r*out.col_size_ + c;
        out.mat_[idx] = PoolingFeature(in_data, kernel_size, stride, type, ch, r, c);
      }
    }
  }
  return out;
}

float Math::PoolingFeature(const Resnet::Matrix3 &in_data, int kernel_size, int stride, std::string type, int ch,
                             int row, int col) {
  //get matrix2
  Matrix2 mat(kernel_size,kernel_size);
  //init mat
  for(int r = 0; r < kernel_size; r++) {
    for(int c = 0; c < kernel_size; c++) {
      int idx = ch*in_data.row_size_*in_data.col_size_ + (row*stride+r)*in_data.col_size_ + (col*stride+c);
      mat.mat_[r][c] = in_data.mat_[idx];
    }
  }

  float value = 0;
  if(type == "MAX") {
    for(int r = 0; r < kernel_size; r++) {
      for(int c = 0; c < kernel_size; c++) {
        if(value <= mat.mat_[r][c]) {
          value = mat.mat_[r][c];
        }
      }
    }
  } else if(type == "AVE") {
    for(int r = 0; r < kernel_size; r++) {
      for(int c = 0; c < kernel_size; c++) {
        value += mat.mat_[r][c];
      }
    }
    value /= (kernel_size*kernel_size);
  }
  return value;
}

Matrix3 Math::ReLU(const Resnet::Matrix3 &in_data, float value) {
  Matrix3 out(in_data.channel_size_, in_data.row_size_, in_data.col_size_, 0);
  for(int ch = 0; ch < out.channel_size_; ch++) {
    for(int row = 0; row < out.row_size_; row++) {
      for(int col = 0; col < out.col_size_; col++) {
        int o_idx = ch*out.row_size_*out.col_size_ + row*out.col_size_ + col;
        int i_idx = ch*in_data.row_size_*in_data.col_size_ + row*in_data.col_size_ + col;
        out.mat_[o_idx] = in_data.mat_[i_idx] > value ? in_data.mat_[i_idx] : value;
      }
    }
  }
  return out;
}

Matrix3 Math::FullConnect(const Resnet::Matrix3 &in_data, const Resnet::InnerProduct &inner) {
  Matrix3 bias(in_data.channel_size_, inner.output_size, 1);
  Matrix3 mat_reshape = Reshape(in_data, 1,in_data.row_size_*in_data.col_size_*in_data.channel_size_,1);
  Matrix3 kernel(1, inner.output_size, mat_reshape.row_size_);
  Matrix3 out = kernel * mat_reshape + bias;
  return out;
}

Matrix3 Math::Reshape(const Resnet::Matrix3 &in_data, int ch, int row, int col) {
  Matrix3 out(ch,row,col,0);
  int count=0;
  for(int i = 0; i < in_data.size_; i++) {
    out.mat_[i] = in_data.mat_[i];
  }
  return out;
}

Matrix3 Math::Softmax(const Resnet::Matrix3 &in_data) {
  Matrix3 temp = in_data;
  //calc temp - max element
  float max_value = -1;
  for(int i = 0; i < in_data.size_; i++) {
    if(max_value < in_data.mat_[i]) {
      max_value = in_data.mat_[i];
    }
  }
  temp -= max_value;
  //calc exp
  Matrix3 temp_exp = CalcExp(temp);
  //calc sum
  float sum = 0;
  for(int i = 0; i < temp_exp.size_; i++) {
    sum += temp_exp.mat_[i];
  }

  //div
  Matrix3 out = temp_exp / sum;
  return out;
}

Matrix3 Math::CalcExp(const Resnet::Matrix3 &in_data) {
  Matrix3 out(in_data.channel_size_, in_data.row_size_, in_data.col_size_, 0);
  for(int i = 0; i < out.size_; i++) {
    out.mat_[i] = exp(in_data.mat_[i]);
    if(out.mat_[i] == 0) {
      out.mat_[i] = (std::numeric_limits<float>::min)();
    }
  }
  return out;
}
} // namespace Resnet