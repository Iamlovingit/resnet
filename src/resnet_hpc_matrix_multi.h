//
// Created by bitfrog on 11/1/19.
//

#include <iostream>
#include <cstdlib>
#include <immintrin.h>
#include <thread>
namespace Resnet {
class MatrixHPC{
 private:
  double **_Matrix;
  size_t _Row,_Column;
 public:
  MatrixHPC():_Matrix(nullptr),_Row(0),_Column(0){}//默认构造
  MatrixHPC(size_t r,size_t c):_Row(r),_Column(c){//构造r行、c列的矩阵
    if(!_Column||!_Row) return;
    _Matrix=(double**)malloc(_Column*sizeof(double*));
    double **p=_Matrix,**end=_Matrix+_Column;
    do{
      *(p++)=(double*)malloc(_Row*sizeof(double));
    }while(p!=end);
  }
  MatrixHPC(size_t r,size_t c,const double init):_Row(r),_Column(c){//构造r行、c列的矩阵并用init初始化
    if(!_Column||!_Row) return;
    _Matrix=(double**)malloc(_Column*sizeof(double*));
    double **pr=_Matrix,**endr=_Matrix+_Column,*p,*end;
    do{
      p=*(pr++)=(double*)malloc(_Row*sizeof(double));
      end=p+_Row;
      do{
        *(p++)=init;
      }while(p!=end);
    }while(pr!=endr);
  }
  MatrixHPC(const MatrixHPC& B){//拷贝构造
    _Row=B._Row;
    _Column=B._Column;
    _Matrix=(double**)malloc(_Column*sizeof(double*));
    double **pbr=B._Matrix,**endbr=B._Matrix+_Column,*pb,*endb,
      **par=_Matrix,**endar=_Matrix+_Column,*pa,*enda;
    do{
      pa=*(par++)=(double*)malloc(_Row*sizeof(double));
      enda=pa+_Row;
      pb=*(pbr++);
      endb=pb+_Row;
      do{
        *(pa++)=*(pb++);
      }while(pa!=enda);
    }while(par!=endar);
  }
  ~MatrixHPC(){//析构
    if(!_Matrix) return;
    double **p=_Matrix,**end=_Matrix+_Column;
    do{
      free(*(p++));
    }while(p!=end);
    _Column=_Row=0;
    free(_Matrix);
  }
  double& operator()(size_t i,size_t j){return _Matrix[j][i];}//访问第i行、第j列的元素
  const double operator()(size_t i,size_t j)const{return _Matrix[j][i];}//访问第i行、第j列的元素
  MatrixHPC& operator=(MatrixHPC&& B){//移动赋值
    if(_Matrix){
      double **p=_Matrix,**end=_Matrix+_Row;
      do{
        free(*(p++));
      }while(p!=end);
      free(_Matrix);
    }
    _Row=B._Row;
    _Column=B._Column;
    _Matrix=B._Matrix;
    B._Matrix=nullptr;
    return *this;
  }

  void multikernel(double **c,double **a,double **b,int row,int col);

  MatrixHPC multi(const MatrixHPC &B);
};
}

