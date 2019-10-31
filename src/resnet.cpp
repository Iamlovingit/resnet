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
  std::cout << "finish load config." << std::endl;

  std::vector<Resnet::Matrix3> input_data;
  Resnet::Matrix3 data(config.input_layer_.channel_size,\
                  config.input_layer_.height_size, \
                  config.input_layer_.width_size);
  input_data.push_back(data);
  std::cout << "main:: inputdata size = " << input_data.size() << std::endl;

  Resnet::FrontPropagation front(input_data,config);
  Poco::Timestamp start;
  front.Train();
  Poco::Timestamp::TimeDiff diff = start.elapsed();
  std::cout << "train cost:" << diff/1000000 << "." << (diff%1000000)/1000 << "s" << std::endl;
  return 0;
}