#ifndef FMOE_UTILS_H
#define FMOE_UTILS_H

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CEIL(_x_,_y_) (((_x_)-1)/(_y_)+1)

#endif  // FMOE_UTILS_H
