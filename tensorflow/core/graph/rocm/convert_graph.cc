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
#include "rocm/include/rtg/program.hpp"

#include "tensorflow/core/graph/algorithm.h"
#include "convert_graph.h"
#include "dump_graph.h"
#include <stack>
#include <unordered_map>

#define RTGLIB tensorflow::rtglib

namespace tensorflow {
namespace rtglib {
namespace convert {

Status ConvertPlaceholder(Converter& ctx, const NodeDef& node_def) {

    return Status::OK();
}

Status ConvertConv2D(Converter& ctx, const NodeDef& node_def) {
    return Status::OK();
}

Status ConvertRelu(Converter& ctx, const NodeDef& node_def) {
    return Status::OK();
}

Status ConvertMaxPool(Converter& ctx, const NodeDef& node_def) {
    return Status::OK();
}

Status ConvertBiasAdd(Converter& ctx, const NodeDef& node_def) {
    return Status::OK();
}

Status ConvertConst(Converter& ctx, const NodeDef& node_def) {
    return Status::OK();
}

Status ConvertIdentity(Converter& ctx, const NodeDef& node_def) {
    return Status::OK();
}

Status ConvertActivation(Converter& ctx, const NodeDef& node_def) {
    return Status::OK();
}

Status ConvertScale(Converter& ctx, const NodeDef& node_def) {
    return Status::OK();
}

void Converter::Register_op_converters()  {
    op_registry_["Placeholder"] = ConvertPlaceholder;
    op_registry_["Conv2D"] = ConvertConv2D;
    op_registry_["Relu"] = ConvertActivation;
    op_registry_["MaxPool"] = ConvertMaxPool;
    op_registry_["BiasAdd"] = ConvertScale;
    op_registry_["Const"] = ConvertConst;
    op_registry_["Identity"] = ConvertIdentity;
}

bool Converter::IsRegistered(Node * node) {
    return op_registry_.count(node->type_string());
}

bool Converter::IsSegmentCandidate(Node * node) {
    return IsRegistered(node);
}
    
Status ConvertGraphToRTG(std::unique_ptr<Graph>* g) {
    CHECK_NOTNULL(g);
    const Graph& graph = **g;
    RTGLIB::dump_graph::DumpGraphToFile("Before convert graph to RTG", graph);
    Node * sinkNode = graph.sink_node();
    CHECK_NOTNULL(sinkNode);
    std::stack<const tensorflow::Node*> iter_stack;
    for (const tensorflow::Edge* edge : sinkNode->in_edges()) {
        iter_stack.push(edge->src());
    }
    Converter conv;
    while (!iter_stack.empty()) {
        const Node *curNode = iter_stack.top();
        iter_stack.pop();
    }

    return Status::OK();
}
    
} // namspace convert
} // namespace rtglib
} // namespace tensorflow

#endif // TENSORFLOW_USE_ROCM

