#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
#error rock-chip triton backend support rv1126 and rk3588 for now!
#elif defined(__aarch64__) || defined(_M_ARM64)
#elif defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__)
#else 
#error unsupported device!
#endif

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>

#include "rknn_api.h"

const char *getBuild() { //Get current architecture, detectx nearly every architecture. Coded by Freak

        #if defined(__x86_64__) || defined(_M_X64)
        return "x86_64";
        #elif defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
        return "x86_32";
        #elif defined(__ARM_ARCH_2__)
        return "ARM2";
        #elif defined(__ARM_ARCH_3__) || defined(__ARM_ARCH_3M__)
        return "ARM3";
        #elif defined(__ARM_ARCH_4T__) || defined(__TARGET_ARM_4T)
        return "ARM4T";
        #elif defined(__ARM_ARCH_5_) || defined(__ARM_ARCH_5E_)
        return "ARM5"
        #elif defined(__ARM_ARCH_6T2_) || defined(__ARM_ARCH_6T2_)
        return "ARM6T2";
        #elif defined(__ARM_ARCH_6__) || defined(__ARM_ARCH_6J__) || defined(__ARM_ARCH_6K__) || defined(__ARM_ARCH_6Z__) || defined(__ARM_ARCH_6ZK__)
        return "ARM6";
        #elif defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__)
        return "ARM7";
        #elif defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__)
        return "ARM7A";
        #elif defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__)
        return "ARM7R";
        #elif defined(__ARM_ARCH_7M__)
        return "ARM7M";
        #elif defined(__ARM_ARCH_7S__)
        return "ARM7S";
        #elif defined(__aarch64__) || defined(_M_ARM64)
        return "ARM64";
        #elif defined(mips) || defined(__mips__) || defined(__mips)
        return "MIPS";
        #elif defined(__sh__)
        return "SUPERH";
        #elif defined(__powerpc) || defined(__powerpc__) || defined(__powerpc64__) || defined(__POWERPC__) || defined(__ppc__) || defined(__PPC__) || defined(_ARCH_PPC)
        return "POWERPC";
        #elif defined(__PPC64__) || defined(__ppc64__) || defined(_ARCH_PPC64)
        return "POWERPC64";
        #elif defined(__sparc__) || defined(__sparc)
        return "SPARC";
        #elif defined(__m68k__)
        return "M68K";
        #else
        return "UNKNOWN";
        #endif
    }
static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
  std::cout<<std::flush;
}

inline bool verifyInputModelInput(rknn_tensor_attr* modelinput,const char* input,int request_count,size_t input_buffer_byte_size){
  /**
   * @brief todo verify model configration and input tensor,
   * etc. shape/nchw/bt.709
   * notice that here we use NCHW.
   */
  uint i=0;
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,(std::string("========================\n")+std::to_string(input_buffer_byte_size)).c_str());

  return true;
}

/*
    the tensor data type.
*/
// typedef enum _rknn_tensor_type {
//     RKNN_TENSOR_FLOAT32 = 0,                            /* data type is float32. */
//     RKNN_TENSOR_FLOAT16,                                /* data type is float16. */
//     RKNN_TENSOR_INT8,                                   /* data type is int8. */
//     RKNN_TENSOR_UINT8,                                  /* data type is uint8. */
//     RKNN_TENSOR_INT16,                                  /* data type is int16. */
//     RKNN_TENSOR_UINT16,                                 /* data type is uint16. */
//     RKNN_TENSOR_INT32,                                  /* data type is int32. */
//     RKNN_TENSOR_UINT32,                                 /* data type is uint32. */
//     RKNN_TENSOR_INT64,                                  /* data type is int64. */
//     RKNN_TENSOR_BOOL,

//     RKNN_TENSOR_TYPE_MAX
// } rknn_tensor_type;
rknn_tensor_type getRKType(TRITONSERVER_DataType tritonType){
  switch (tritonType)
  {
    case TRITONSERVER_TYPE_BOOL:
      return RKNN_TENSOR_BOOL;
    case TRITONSERVER_TYPE_UINT8:
      return RKNN_TENSOR_UINT8;
    case TRITONSERVER_TYPE_UINT16:
      return RKNN_TENSOR_UINT16;
    case TRITONSERVER_TYPE_UINT32:
      return RKNN_TENSOR_UINT32;
    case TRITONSERVER_TYPE_UINT64:
      return RKNN_TENSOR_INT64;
    case TRITONSERVER_TYPE_INT8:
      return RKNN_TENSOR_INT8;
    case TRITONSERVER_TYPE_INT16:
      return RKNN_TENSOR_INT16;
    case TRITONSERVER_TYPE_INT32:
      return RKNN_TENSOR_INT32;
    case TRITONSERVER_TYPE_INT64:
      return RKNN_TENSOR_INT64;
    case TRITONSERVER_TYPE_FP16:
      return RKNN_TENSOR_FLOAT16;
    case TRITONSERVER_TYPE_FP32:
      return RKNN_TENSOR_FLOAT32;
    case TRITONSERVER_TYPE_FP64:
      return RKNN_TENSOR_TYPE_MAX;
    case TRITONSERVER_TYPE_BYTES:
      return RKNN_TENSOR_UINT8;
    case TRITONSERVER_TYPE_BF16:
      return RKNN_TENSOR_FLOAT16;
    default:
      break;
  }
  return RKNN_TENSOR_UINT8;
}


TRITONSERVER_Error*
DimsJsonToDimVec(
    triton::common::TritonJson::Value& dims_json, std::vector<int64_t>* dims)
{
  dims->clear();
  for (size_t i = 0; i < dims_json.ArraySize(); i++) {
    int64_t dim;
    RETURN_IF_ERROR(dims_json.IndexAsInt(i, &dim));
    dims->push_back(dim);
  }
  return nullptr;
}

  // TRITONSERVER_TYPE_INVALID,
  // TRITONSERVER_TYPE_BOOL,
  // TRITONSERVER_TYPE_UINT8,
  // TRITONSERVER_TYPE_UINT16,
  // TRITONSERVER_TYPE_UINT32,
  // TRITONSERVER_TYPE_UINT64,
  // TRITONSERVER_TYPE_INT8,
  // TRITONSERVER_TYPE_INT16,
  // TRITONSERVER_TYPE_INT32,
  // TRITONSERVER_TYPE_INT64,
  // TRITONSERVER_TYPE_FP16,
  // TRITONSERVER_TYPE_FP32,
  // TRITONSERVER_TYPE_FP64,
  // TRITONSERVER_TYPE_BYTES,
  // TRITONSERVER_TYPE_BF16
TRITONSERVER_DataType getTritonDT(std::string xx){
  if(!xx.compare("TYPE_UINT8"))
    return TRITONSERVER_TYPE_UINT8;
  else if (!xx.compare("TYPE_UINT16"))
    return TRITONSERVER_TYPE_UINT16;
  else if (!xx.compare("TYPE_UINT32"))
    return TRITONSERVER_TYPE_UINT32;
  else if (!xx.compare("TYPE_UINT64"))
    return TRITONSERVER_TYPE_UINT64;
  else if (!xx.compare("TYPE_INT8"))
    return TRITONSERVER_TYPE_INT8;
  else if (!xx.compare("TYPE_INT16"))
    return TRITONSERVER_TYPE_INT16;
  else if (!xx.compare("TYPE_INT3"))
    return TRITONSERVER_TYPE_INT32;
  else if (!xx.compare("TYPE_INT64"))
    return TRITONSERVER_TYPE_INT64;
  else if (!xx.compare("TYPE_FP16"))
    return TRITONSERVER_TYPE_FP16;
  else if (!xx.compare("TYPE_FP32"))
    return TRITONSERVER_TYPE_FP32;
  else if (!xx.compare("TYPE_FP64"))
    return TRITONSERVER_TYPE_FP64;
  else if (!xx.compare("TYPE_BYTES"))
    return TRITONSERVER_TYPE_BYTES;

  return TRITONSERVER_TYPE_UINT8;
}

