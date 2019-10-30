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
      Matrix3 in_data = map_progress[layer.bottom[0]];
      Matrix3 kernel(layer.convolution.output_size * in_data.channel_size_,
                     layer.convolution.kernel_size,
                     layer.convolution.kernel_size);
      Matrix3 out = Math::Conv(in_data, kernel, layer.convolution);
      map_progress[layer.name] = out;
    } else if (layer.type == "\"BatchNorm\"") {
      Matrix3 in_data = map_progress[layer.bottom[0]];
      Matrix3 out = Math::BN(in_data,
                             layer.batch_norm.moving_average_fraction,
                             layer.batch_norm.eps,
                             layer.batch_norm.scale_bias);
      map_progress[layer.name] = out;
    } else if (layer.type == "\"Pooling\"") {
      Matrix3 in_data = map_progress[layer.bottom[0]];
      Matrix3 out = Math::Pooling(in_data, layer.pooling.kernel_size, layer.pooling.stride, layer.pooling.pool);
      map_progress[layer.name] = out;
    } else if (layer.type == "\"Eltwise\"") {
      Matrix3 in_data1 = map_progress[layer.bottom[0]];
      Matrix3 in_data2 = map_progress[layer.bottom[1]];
      Matrix3 out = in_data1 + in_data2;
      map_progress[layer.name] = out;
    } else if (layer.type == "\"InnerProduct\"") {
      Matrix3 in_data = map_progress[layer.bottom[0]];
      Matrix3 out = Math::FullConnect(in_data, layer.inner_product);
      map_progress[layer.name] = out;
    } else if (layer.type == "\"Softmax\"") {
      Matrix3 in_data = map_progress[layer.bottom[0]];
      Matrix3 out = Math::Softmax(in_data);
      map_progress[layer.name] = out;
    } else if (layer.type == "\"ReLU\"") {
      Matrix3 in_data = map_progress[layer.bottom[0]];
      Matrix3 out = Math::ReLU(in_data, 0.0);
      map_progress[layer.name] = out;
    }
  }
  return 0;
}
} // namespace Resnet