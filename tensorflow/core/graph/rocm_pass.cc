/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class ROCMPass : public GraphOptimizationPass {
  public:
    ROCMPass() {}
    
    // Standard interface to run pass
    Status Run(const GraphOptimizationPassOptions& options);
};

status ROMCPass::Run(
  const GraphOptimizationPassOptions& options) {
    return Status::OK();
}
                         
    
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 1, ROCMPass);
    

} // namespace tensorflow

#endif // TENSORFLOW_USE_ROCM
