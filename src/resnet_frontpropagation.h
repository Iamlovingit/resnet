//
// Created by bitfrog on 10/28/19.
//

#pragma once
#include "resnet_config.h"
#include "resnet_matrix.h"
#include "resnet_math.h"
namespace Resnet {
class FrontPropagation {
 public:
  FrontPropagation(Matrix3,Config);
  int Train();
 private:
  Matrix3 in_data_;
  Config config_;
  std::map<std::string, Matrix3> map_progress;
};
}
