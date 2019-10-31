//
// Created by bitfrog on 10/28/19.
//

#include "resnet_frontpropagation.h"
namespace Resnet {
FrontPropagation::FrontPropagation(std::vector<Resnet::Matrix3>& input_data, Resnet::Config config) {
  in_data_ = input_data;
  std::cout << "construct in_data_ size =" << in_data_.size() << std::endl;
  config_ = config;
}

int FrontPropagation::Train() {
  //init input layer.
  map_progress["\"data\""] = in_data_;
  std::vector<Matrix3> test = map_progress["\"data\""];
  for(int i = 0; i < config_.vec_layer_.size(); i++) {
    std::cout << "train process: " << i << "/" << config_.vec_layer_.size() << std::endl;
    Layer layer = config_.vec_layer_[i];
    if(layer.type == "\"Convolution\"") {
      //get input data .
      std::vector<Matrix3> in_data = map_progress[layer.bottom[0]];
      std::cout << "layer.bottom name =" << layer.bottom[0] << std::endl;
      std::cout << "in_data size = " << in_data.size() << std::endl;
      std::vector<Matrix3> out;
      for(auto it : in_data) {
        Matrix3 kernel(layer.convolution.output_size * it.channel_size_,
                       layer.convolution.kernel_size,
                       layer.convolution.kernel_size);
        std::cout << "push back! conv out" << std::endl;
        out.push_back(Math::Conv(it, kernel, layer.convolution));
      }
      std::cout << layer.name << " out.size = " << out.size() <<std::endl;
      map_progress[layer.name] = out;
    } else if (layer.type == "\"BatchNorm\"") {
      std::vector<Matrix3> in_data = map_progress[layer.bottom[0]];
      std::vector<Matrix3> out = Math::BN(in_data,
                             layer.batch_norm.moving_average_fraction,
                             layer.batch_norm.eps,
                             layer.batch_norm.scale_bias);
      map_progress[layer.name] = out;
    } else if (layer.type == "\"Pooling\"") {
      std::vector<Matrix3> in_data = map_progress[layer.bottom[0]];
      std::vector<Matrix3> out;
      for(auto it : in_data) {
        out.push_back(Math::Pooling(it, layer.pooling.kernel_size, layer.pooling.stride, layer.pooling.pool));
      }
      map_progress[layer.name] = out;
    } else if (layer.type == "\"Eltwise\"") {
      std::vector<Matrix3> in_data1 = map_progress[layer.bottom[0]];
      std::vector<Matrix3> in_data2 = map_progress[layer.bottom[1]];
      std::vector<Matrix3> out;
      for(int i = 0; i < in_data1.size(); i++) {
        out.push_back(in_data1[i] + in_data2[i]);
      }
      map_progress[layer.name] = out;
    } else if (layer.type == "\"InnerProduct\"") {
      std::vector<Matrix3> in_data = map_progress[layer.bottom[0]];
      std::vector<Matrix3> out;
      for(auto it : in_data) {
        out.push_back(Math::FullConnect(it, layer.inner_product));
      }
      map_progress[layer.name] = out;
    } else if (layer.type == "\"Softmax\"") {
      std::vector<Matrix3> in_data = map_progress[layer.bottom[0]];
      std::vector<Matrix3> out;
      for(auto it : in_data) {
        out.push_back(Math::Softmax(it));
      }
      map_progress[layer.name] = out;
    } else if (layer.type == "\"ReLU\"") {
      std::vector<Matrix3> in_data = map_progress[layer.bottom[0]];
      std::vector<Matrix3> out;
      for(auto it : in_data) {
        out.push_back(Math::ReLU(it, 0.0));
      }
      map_progress[layer.name] = out;
    }
  }
  std::cout << "Train over." << std::endl;
  return 0;
}
} // namespace Resnet