//
// Created by bitfrog on 10/28/19.
//

#include "resnet_math.h"

namespace  Resnet {
Matrix3 Math::Conv(const Matrix3 &in_data, const Matrix3 &kernel, Convolution conv_para) {
  Matrix3 pad_m = PadMatrix(in_data, conv_para.pad); //padding input matrix
  int out_size = (in_data.row_size_ + 2*conv_para.pad - conv_para.kernel_size)/2+1;
  Matrix3 out(conv_para.output_size, out_size, out_size, 0); // init out matrix
  //value every element.
  for(int ch = 0; ch < out.channel_size_; ch++) {
    for(int c = 0; c < in_data.channel_size_; c++) { // every input channel
       for(int row = 0; row < out.row_size_; row++) { //every row
         for(int col = 0; col < out.col_size_; col++) { //every col
           out.mat_[ch][row][col] += Feature(pad_m, kernel, ch, c, row, col, conv_para.stride);
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
        if(row < pad || col < pad || row > out.row_size_-pad-1 || col > out.col_size_-pad-1) {
          out.mat_[ch][row][col] = 0;
        } else {
          out.mat_[ch][row][col] = in_data.mat_[ch][row-pad][col-pad];
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
      data_src.mat_[i][j] = data.mat_[in_channel_no][row*stride+i][col*stride+j];
    }
  }

  //init ker_src
  for(int i = 0; i < data_src.row_size_; i++) {
    for(int j = 0; j < data_src.col_size_; j++) {
      ker_src.mat_[i][j] = kernel.mat_[channel_no*data.channel_size_+channel_no][i][j];
    }
  }

  //calc value
  for(int i = 0; i < data_src.row_size_; i++) {
    for(int j = 0; j < data_src.col_size_; j++)
      value += data_src.mat_[i][j]*ker_src.mat_[i][j];
  }
  return value;
}
} // namespace Resnet