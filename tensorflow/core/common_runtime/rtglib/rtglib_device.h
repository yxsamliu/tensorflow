/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if !TENSORFLOW_USE_RTGLIB
#error This file must only be included when building TensorFlow with RTGLIB support
#endif

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_RTGLIB_RTGLIB_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_RTGLIB_RTGLIB_DEVICE_H_

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/rtglib/rtglib_allocator.h"
#include "tensorflow/core/common_runtime/rtglib/rtglib_device_context.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class GRTGLIBInterface
{
    std::vector<Allocator*>                 m_cpu_allocator_;      // not owned
    std::vector<RTGLIBAllocator*>             m_rtglib_allocator_;     // owned
    std::vector<RTGLIBDeviceContext*>         m_rtglib_context_;       // owned

    static std::mutex mutex_;
    static GRTGLIBInterface* s_instance;
    GRTGLIBInterface() {
      m_cpu_allocator_.push_back(cpu_allocator());
      m_rtglib_allocator_.push_back(new RTGLIBAllocator());
      m_rtglib_context_.push_back(new RTGLIBDeviceContext());
    }

    ~GRTGLIBInterface() {
        m_cpu_allocator_.clear();
        for (auto p : m_rtglib_allocator_) {
          p->Synchronize();
          delete p;
        }
        m_rtglib_allocator_.clear();

        for(auto p : m_rtglib_context_) {
          p->Unref();
        }
        m_rtglib_context_.clear();
    }

  public:
    static GRTGLIBInterface *instance()
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!s_instance) {
        s_instance = new GRTGLIBInterface();
      }
      return s_instance;
    }

    static void Reset()
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if(s_instance) {
        delete s_instance;
        s_instance = NULL;
      }
    }

    RTGLIBAllocator * GetRTGLIBAllocator(size_t i = 0) {
      if(!m_rtglib_allocator_.empty()) {
        return m_rtglib_allocator_[i];
      } else {
        std::cerr << "No RTGLIB device has been added" << std::endl;
        return nullptr;
      }
    }

    Allocator * GetCPUAllocator(size_t i = 0) {
      if(!m_cpu_allocator_.empty()) {
        return m_cpu_allocator_[i];
      } else {
        std::cerr << "No RTGLIB device has been added" << std::endl;
        return nullptr;
      }
    }

    RTGLIBDeviceContext * GetRTGLIBContext(size_t i = 0) {
      if(!m_rtglib_context_.empty()) {
        return m_rtglib_context_[i];
      } else {
        std::cerr << "No RTGLIB device has been added" << std::endl;
        return nullptr;
      }
    }

    string GetShortDeviceDescription(int device_id = 0) {
      return strings::StrCat("RTGLIB:", device_id);
    }
};

class RTGLIBDevice : public LocalDevice {
 public:
  RTGLIBDevice(const SessionOptions &options, const string &name,
             Bytes memory_limit, const DeviceLocality &locality,
             const string &physical_device_desc, RTGLIBAllocator * rtglib_allocator,
             Allocator *cpu_allocator, RTGLIBDeviceContext* ctx)
      : LocalDevice(
            options,
            Device::BuildDeviceAttributes(name, DEVICE_RTGLIB, memory_limit,
                                          locality, physical_device_desc)),
        cpu_allocator_(cpu_allocator),
        rtglib_allocator_(rtglib_allocator),
        device_context_(ctx) {
    RegisterDevice();
  }

  ~RTGLIBDevice() override;

  void Compute(OpKernel *op_kernel, OpKernelContext *context) override;
  Allocator *GetAllocator(AllocatorAttributes attr) override;
  Status MakeTensorFromProto(const TensorProto &tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor *tensor) override;

  Status FillContextMap(const Graph *graph,
                        DeviceContextMap *device_context_map) override;

  Status Sync() override;

 private:
  void RegisterDevice();

  Allocator         *cpu_allocator_;           // not owned
  RTGLIBAllocator     *rtglib_allocator_;          // not owned
  RTGLIBDeviceContext *device_context_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_RTGLIB_RTGLIB_DEVICE_H_
