//
// Created by bitfrog on 10/23/19.
//

#include <fstream>
#include <string>
#include <iostream>

#include "resnet_config.h"
#include "resnet_3rd.h"
namespace Resnet {
bool Config::Load() {
  std::ifstream ifile;
  ifile.open(file_name_.c_str());
  std::string lbuf;
  while(getline(ifile,lbuf)) {
    lbuf = Poco::trim(lbuf);
    if(lbuf[0] == '#') continue;
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
    if(file_buf_[i].length() > 0) {
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
    Layer layer;
    //find layer head.
    if(file_buf_[cur].find("layer") != std::string::npos) {
      cur++; // move to layer body line
      std::string lbuf = file_buf_[cur++]; //name line
      Poco::StringTokenizer name_st(lbuf,":");
      layer.name = name_st[1];
      lbuf = file_buf_[cur++]; // type line
      Poco::StringTokenizer type_st(lbuf,":");
      layer.type = type_st[1];
      lbuf = file_buf_[cur++]; // bottom line
      Poco::StringTokenizer bottom_st(lbuf,":");
      layer.bottom = bottom_st[1];
      lbuf = file_buf_[cur++]; // top line
      Poco::StringTokenizer top_st(lbuf,":");
      layer.top = top_st[1];

      if(layer.type == "\"Convolution\"") {
        cur++; //pass over head line.
        while(1) { // init convolution body
          lbuf = file_buf_[cur++]; // get line.
          if(lbuf == "}") break;
          Poco::StringTokenizer st(lbuf,":");
          if(st[0] == "num_output") { // num_output line
            layer.convolution.output_size = atoi(st[1].c_str());
          } else if(st[0] == "kernel_size") { // kernel size
            layer.convolution.kernel_size = atoi(st[1].c_str());
          } else if (st[0] == "pad") { //pad line
            layer.convolution.pad = atoi(st[1].c_str());
          } else if (st[0] == "stride") { //stride line
            layer.convolution.stride = atoi(st[1].c_str());
          } else if (st[0] == "bias_term") { // bias_term line
            layer.convolution.bias_term = (st[1]=="true"?true:false);
          } else if (lbuf.find("weight_filler") != std::string::npos) {
            cur++; //init weight
            std::string ws = file_buf_[cur++];
            Poco::StringTokenizer wst(ws, ":");
            layer.convolution.weight_filler.type = wst[1];
            cur++; //pass over weight "}"
          }
        }
        cur++; //pass over convolution "}"
      } else if(layer.type == "\"BatchNorm\"") { //batch norm
        cur++;
        while(1) {
          lbuf = file_buf_[cur++]; // get line.
          if(lbuf == "}") break;
          Poco::StringTokenizer st(lbuf,":");
          if(st[0] == "moving_average_fraction") {
            layer.batch_norm.moving_average_fraction = atof(st[1].c_str());
          } else if(st[0] == "eps") {
            layer.batch_norm.eps = atof(st[1].c_str());
          } else if (st[0] == "scale_bias") {
            layer.batch_norm.scale_bias = (st[1]=="true"?true:false);
          }
        }
        cur++; //pass over batch norm "}"
      } else if (layer.type == "\"Pooling\"") {
        cur++;
        while(1) {
          lbuf = file_buf_[cur++]; //get line
          if(lbuf == "}") break;
          Poco::StringTokenizer st(lbuf,":");
          if(st[0] == "pool") {
            layer.pooling.pool = st[1];
          } else if(st[0] == "kernel_size") {
            layer.pooling.kernel_size = atoi(st[1].c_str());
          } else if(st[0] == "stride") {
            layer.pooling.stride = atoi(st[1].c_str());
          }
        }
        cur++; //pass over pooling_pram "}"
      } else if (layer.type == "\"Eltwise\"") {
        cur++;
        Poco::StringTokenizer st(file_buf_[cur++], ":");
        layer.elt_wise.operation = st[1];
        cur++;//pass over elt_wise "}"
      } else if (layer.type == "\"InnerProduct\"") {
        cur++;
        while(1) {
          lbuf = file_buf_[cur++];
          if(lbuf == "}") break;
          Poco::StringTokenizer st(lbuf,":");
          if(st[0] == "num_output") {
            layer.inner_product.output_size = atoi(st[1].c_str());
          } else if (lbuf.find("weight_filler") != std::string::npos) {
            cur++;
            std::string ws = file_buf_[cur++];
            Poco::StringTokenizer wst(ws, ":");
            layer.inner_product.weight_filler.type = wst[1];
            cur++;
          } else if (lbuf.find("bias_filler") != std::string::npos) {
            while(1) {
              cur++;
              std::string bs = file_buf_[cur++];
              if(bs == "}") break;
              Poco::StringTokenizer bst(bs, ":");
              if(bst[0] == "type") {
                layer.inner_product.bias_filler.type = bst[1];
              } else if(bst[0] == "value") {
                layer.inner_product.bias_filler.value = atoi(bst[1].c_str());
              }
            }
            cur++; //pass over bias filler "}"
          }
        }
        cur++; //pass over inner_product "}"
      }
      cur++; //pass over layer "}"
    }
    map_layer_[layer.name] = layer;
    vec_layer_.push_back(layer);
  }
  return cur;
}

void Config::operator=(Resnet::Config config) {
  this->file_buf_ = config.file_buf_;
  this->file_name_ = config.file_name_;
  this->total_line_no_ = config.total_line_no_;
  this->cur_line_no_ = config.cur_line_no_;

  this->input_layer_ = config.input_layer_;
  this->map_layer_ = config.map_layer_;
  this->vec_layer_ = config.vec_layer_;
}
}