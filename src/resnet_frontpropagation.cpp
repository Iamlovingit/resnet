//
// Created by bitfrog on 10/28/19.
//

#include "resnet_frontpropagation.h"
namespace Resnet {
FrontPropagation::FrontPropagation(Resnet::Matrix input_data, Resnet::Config config) {
  in_data_ = input_data;
  config_ = config;
}

int FrontPropagation::Train() {
  //init input layer.

  for(int i = 0; i < config_.vec_layer_.size(); i++) {

  }
}
} // namespace Resnet