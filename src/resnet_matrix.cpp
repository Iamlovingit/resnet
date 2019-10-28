//
// Created by bitfrog on 10/28/19.
//

#include "resnet_matrix.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
namespace Resnet {
#define MAX_RAND 999
Matrix::Matrix(int row_size, int col_size, int channel_size) {
  srand(time(NULL));
  mat_ = (float***) malloc(channel_size * sizeof(float**));
  for(int i = 0; i < channel_size; i++) {
    mat_[i] = (float**) malloc(row_size * sizeof(float*));
    for(int j = 0; j < row_size; j++) {
      mat_[i][j] = (float*) malloc(col_size * sizeof(float));
      for(int k = 0; k < col_size; k++) {
        mat_[i][j][k] = rand()%(MAX_RAND+1)/(float)(MAX_RAND+1); // random (0-1) float value.
      }
    }
  }

}
} //namespace Resnet