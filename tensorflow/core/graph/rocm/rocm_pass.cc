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
class ROCmPass : public GraphOptimizationPass {
  public:
    ROCmPass() {}
    
    // Standard interface to run pass
    Status Run(const GraphOptimizationPassOptions& options);
};

const OptimizationPassRegistry::Grouping kROCmPassGroup =
    OptimizationPassRegistry::POST_PARTITIONING;
    
REGISTER_OPTIMIZATION(kROCmPassGroup, 1, ROCmPass);    
Status ROCmPass::Run(
  const GraphOptimizationPassOptions& options) {
    if (options.graph == nullptr && options.partition_graphs == nullptr) {
        return Status::OK();
    }
    if (kROCmPassGroup != OptimizationPassRegistry::POST_PARTITIONING) {
        // For any pre-partitioning phase, a graph is stored in options.graph.
        /// ConvertGraphToRTG(options.graph);
    } else {
        // For post partitioning phase, graphs are stored in
        // options.partition_graphs.
        for (auto& pg : *options.partition_graphs) {
            // ConvertGraphToRTG(&pg.second);
        }
    }
    
    return Status::OK();
}
                         
} // namespace tensorflow

#endif // TENSORFLOW_USE_ROCM
