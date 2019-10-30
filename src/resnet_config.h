//
// Created by bitfrog on 10/23/19.
//

#pragma once
#include <vector>
#include <string>
#include <map>
namespace Resnet {
struct BiasFiller {
  std::string type;
  int value;
};

struct WeightFiller {
  std::string type;
};

struct InnerProduct {
  int output_size;
  WeightFiller weight_filler;
  BiasFiller bias_filler;
};

struct EltWise {
  std::string operation;
};

struct Pooling {
  std::string pool;
  int kernel_size;
  int stride;
};

struct BatchNorm {
  float moving_average_fraction;
  float eps;
  bool scale_bias;
};

struct Convolution {
  int output_size;
  int kernel_size;
  int pad;
  int stride;
  WeightFiller weight_filler;
  bool bias_term;
};

struct InputLayer {
  std::string title;
  std::string name;
  int batch_size;
  int channel_size;
  int height_size;
  int width_size;
};


struct Layer {
  std::string name;
  std::string type;
  std::vector<std::string> bottom;
  std::string top;
  Convolution convolution;
  BatchNorm batch_norm;
  Pooling pooling;
  EltWise elt_wise;
  InnerProduct inner_product;
};

class Config {
 public:
  Config() {
    file_name_.clear();
    file_buf_.clear();
    total_line_no_=0;
    cur_line_no_=0;
  }
  explicit Config(std::string file_name):
  file_name_(file_name){
  }
  bool Load();
  void operator=(Config);
 private:
  bool Parse();
  int ParseInputLayer();
  int ParseLayer();

 private:
  std::string file_name_;
  std::vector<std::string> file_buf_;
  int total_line_no_;
  int cur_line_no_;
 public:
  InputLayer input_layer_;
  std::map<std::string, Layer> map_layer_;
  std::vector<Layer> vec_layer_;
};
} // namespace RESNET
