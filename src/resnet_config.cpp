//
// Created by bitfrog on 10/23/19.
//

#include <fstream>
#include <string>
#include <iostream>

#include "resnet_config.h"
#include "resnet_3rd.h"
namespace RESNET {
bool Config::Load() {
  std::ifstream ifile;
  ifile.open(file_name_.c_str());
  std::string lbuf;
  while(getline(ifile,lbuf)) {
    lbuf = Poco::trim(lbuf);
    file_buf_.push_back(lbuf);
  }
  total_line_no_ = file_buf_.size();
  cur_line_no_ = 0;
  bool ret = Parse();
  return ret;
}

bool Config::Parse() {
  //parse input layer
  cur_line_no_ = ParseInputLayer();
  if(cur_line_no_ < 0) {
    std::cout << "ParseInputLayer failed." << std::endl;
    return false;
  }

  cur_line_no_ = ParseLayer();
  if(cur_line_no_ < 0) {
    std::cout << "ParseLayer failed." << std::endl;
    return false;
  }

  return true;
}

int Config::ParseInputLayer() {
  //ignore empty lines.
  int cur = 0;
  for(int i = 0; i < file_buf_.size(); i++) {
    if(file_buf_[i][0] != '#' && file_buf_[i].length() > 0) {
      break;
    }
    cur++;
  }
  // first line is title
  Poco::StringTokenizer st(file_buf_[cur++],":");
  input_layer_.title = st[1];
  Poco::StringTokenizer st1(file_buf_[cur++], ":");
  input_layer_.name = st1[1];
  Poco::StringTokenizer st2(file_buf_[cur++], ":");
  input_layer_.batch_size = atoi(st2[1].c_str());
  Poco::StringTokenizer st3(file_buf_[cur++], ":");
  input_layer_.channel_size = atoi(st3[1].c_str());
  Poco::StringTokenizer st4(file_buf_[cur++], ":");
  input_layer_.height_size = atoi(st4[1].c_str());
  Poco::StringTokenizer st5(file_buf_[cur++], ":");
  input_layer_.width_size = atoi(st5[1].c_str());

//  std::cout << "title = " << input_layer_.title << std::endl;
//  std::cout << "name = " << input_layer_.name << std::endl;
//  std::cout << "batch_size = " << input_layer_.batch_size << std::endl;
//  std::cout << "channel_size = " << input_layer_.channel_size << std::endl;
//  std::cout << "height_size = " << input_layer_.height_size << std::endl;
//  std::cout << "width_size = " << input_layer_.width_size << std::endl;
//  std::cout << "cur = " << cur << std::endl;
  return cur;
}

int Config::ParseLayer() {
  int cur = cur_line_no_;
  while(cur < file_buf_.size()) {
    //ignore # line.
    if(file_buf_[cur][0] == '#') {
      cur++;
      continue;
    }
    //find layer head.
    if(file_buf_[cur].find("layer") != std::string::npos) {
      cur++; // move to layer body line
      
    }
  }
};
}