//
// Created by bitfrog on 10/23/19.
//

#include "resnet.h"

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Resnet::Config config(FLAGS_config_file);
  bool ret = config.Load();
  if(!ret) {
    std::cout << "Config Load failed." << std::endl;
    return -1;
  }

  Resnet::Matrix input_data(config.input_layer_.channel_size,\
                  config.input_layer_.height_size, \
                  config.input_layer_.width_size);

  Resnet::FrontPropagation front(input_data,config);
  front.Train();
  return 0;
}