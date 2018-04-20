/* 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifdef TENSORFLOW_USE_ROCM

#include "convert_graph.h"
#include "dump_graph.h"

#define RTGLIB tensorflow::rtglib

namespace tensorflow {
namespace rtglib {
namespace convert {

Status ConvertGraphToRTG(std::unique_ptr<Graph>* graph) {
    CHECK_NOTNULL(graph);
    RTGLIB::dump_graph::DumpGraphToFile("Before convert graph to RTG", **graph);
    return Status::OK();
}
    
} // namspace convert
} // namespace rtglib
} // namespace tensorflow

#endif // TENSORFLOW_USE_ROCM

