//
// Created by bitfrog on 10/28/19.
//

#pragma once
#include "resnet_config.h"
#include "resnet_matrix.h"
namespace Resnet {
class FrontPropagation {
 public:
  FrontPropagation(Matrix,Config);
  int Train();
 private:
  Matrix in_data_;
  Config config_;
  std::map<std::string, Matirx> map_progress;
};
}
