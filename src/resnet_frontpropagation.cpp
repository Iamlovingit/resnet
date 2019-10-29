//
// Created by bitfrog on 10/28/19.
//

#include "resnet_frontpropagation.h"
namespace Resnet {
FrontPropagation::FrontPropagation(Resnet::Matrix3 input_data, Resnet::Config config) {
  in_data_ = input_data;
  config_ = config;
}

int FrontPropagation::Train() {
  //init input layer.
  map_progress["data"] = in_data_;
  for(int i = 0; i < config_.vec_layer_.size(); i++) {
    Layer layer = config_.vec_layer_[i];
    if(layer.type == "\"Convolution\"") {
      //get input data .
      Matrix3 in_data = map_progress[layer.bottom];
      Matrix3 kernel(layer.convolution.output_size * in_data.channel_size_,
                     layer.convolution.kernel_size,
                     layer.convolution.kernel_size);
      Matrix3 out = Math::Conv(in_data, kernel, layer.convolution);
      map_progress[layer.name] = out;
    } else if (layer.type == "\"BatchNorm\"") {

    } else if (layer.type == "\"Pooling\"") {

    }
  }
}
} // namespace Resnet