//
// Created by bitfrog on 10/23/19.
//

#include "resnet.h"

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  RESNET::Config config(FLAGS_config_file);
  bool ret = config.Load();
  if(!ret) {
    std::cout << "Config Load failed." << std::endl;
    return -1;
  }

  LoadData();

  Train();

  std::cout << FLAGS_config_file << std::endl;
  std::cout << FLAGS_train_data_file << std::endl;
  return 0;
}