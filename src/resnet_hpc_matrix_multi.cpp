//
// Created by bitfrog on 11/1/19.
//

#include "resnet_hpc_matrix_multi.h"
namespace Resnet {
void MatrixHPC::multikernel(double **c,double **a,double **b,int row,int col){
  __m128d t01_0,t01_1,t01_2,t01_3,t23_0,t23_1,t23_2,t23_3,
    a0,a1,b0,b1,b2,b3;
  t01_0=t01_1=t01_2=t01_3=t23_0=t23_1=t23_2=t23_3=_mm_set1_pd(0);
  double *pb0(b[col]),*pb1(b[col+1]),*pb2(b[col+2]),*pb3(b[col+3]),*pa0(a[0]),*pa1(a[1]),*endb0=pb0+_Column;
  do{
    a0=_mm_load_pd(pa0);
    a1=_mm_load_pd(pa1);
    b0=_mm_set1_pd(*(pb0++));
    b1=_mm_set1_pd(*(pb1++));
    b2=_mm_set1_pd(*(pb2++));
    b3=_mm_set1_pd(*(pb3++));
    t01_0+=a0*b0;
    t01_1+=a0*b1;
    t01_2+=a0*b2;
    t01_3+=a0*b3;
    t23_0+=a1*b0;
    t23_1+=a1*b1;
    t23_2+=a1*b2;
    t23_3+=a1*b3;
    pa0+=2;
    pa1+=2;
  }while(pb0!=endb0);
  _mm_store_pd(&c[col][row],t01_0);
  _mm_store_pd(&c[col+1][row],t01_1);
  _mm_store_pd(&c[col+2][row],t01_2);
  _mm_store_pd(&c[col+3][row],t01_3);
  _mm_store_pd(&c[col][row+2],t23_0);
  _mm_store_pd(&c[col+1][row+2],t23_1);
  _mm_store_pd(&c[col+2][row+2],t23_2);
  _mm_store_pd(&c[col+3][row+2],t23_3);
}

MatrixHPC MatrixHPC::multi(const MatrixHPC &B){
  if(_Column!=B._Row) return *this;
  MatrixHPC tmp(_Row,B._Column,0);
  double *ta[2];
  ta[0]=(double*)malloc(sizeof(double)*2*_Column);
  ta[1]=(double*)malloc(sizeof(double)*2*_Column);

  int i(0),j(0),k,t;
  do{
    k=0;i=0;
    do{
      ta[0][k]=_Matrix[i][j];
      ta[1][k++]=_Matrix[i][j+2];
      ta[0][k]=_Matrix[i][j+1];
      ta[1][k++]=_Matrix[i++][j+3];
    }while(i<_Column);
    i=0;
    do{
      multikernel(tmp._Matrix,ta,B._Matrix,j,i);
      i+=4;
    }while(i<B._Column);
    j+=4;
  }while(j<_Row);

  free(ta[0]);
  free(ta[1]);
  return tmp;
}
}
