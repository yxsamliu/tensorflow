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
#ifndef TENSORFLOW_RTGLIB_CONVERT_
#define TENSORFLOW_RTGLIB_CONVERT_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"

#include "rocm/include/rtg/program.hpp"

namespace tensorflow {
namespace rtglib {
namespace convert {

#define MIN_CLUSTER_SIZE 3    

struct Cluster {
     explicit Cluster() { init(); }
     void addInputEdge(const Edge* edge)  { input_edges.push_back(edge);  }
     void addOutputEdge(const Edge* edge) { output_edges.push_back(edge); }
     void addNode(Node* node)             { nodes.push_back(node);        }
     int  getSize()                       { return nodes.size();          }
     std::vector<const Edge*> input_edges;
     std::vector<const Edge*> output_edges;
     std::vector<Node*> nodes;  // sorted in reversed post order.
     void init() {
         input_edges.clear();
         output_edges.clear();
         nodes.clear();
     }
};

typedef std::vector<rtg::instruction*> T_RTG_INST_V; 
typedef const std::vector<std::pair<string, Tensor>> T_INPUT_MAP; 
 
class  Converter;
using OpConverter =
    std::function<tensorflow::Status(Converter&, const tensorflow::NodeDef&, const T_RTG_INST_V&)>;
    
struct Converter {
    explicit Converter(rtg::program* p, T_INPUT_MAP* map) {
        Init(); program = p; inputs = map;
    }
    bool isParameter(const rtg::instruction&);
    bool isRegistered(const Node*);
    void add_instruction(const Node*);
    void add_parameter(const NodeDef&);
    DataType get_type(const rtg::shape&);
    void get_shape_proto(const rtg::shape&, TensorShapeProto*);
    rtg::shape parse_type(const NodeDef&, DataType&);
    std::unordered_map<string, OpConverter> op_registry_;
    void Init() {
        register_op_converters();
    }
    void register_op_converters();
    bool starts_with(const string& value, const string& prefix);
    std::unordered_map<std::string, rtg::instruction*> instructions;
    rtg::program* program;
    T_INPUT_MAP* inputs;
};

Status AddActivation(Converter&, const NodeDef&, const T_RTG_INST_V&);
Status AddBiasAdd(Converter&, const NodeDef&, const T_RTG_INST_V&);
Status AddConst(Converter&, const NodeDef&, const T_RTG_INST_V&);
Status AddConv2D(Converter&, const NodeDef&, const T_RTG_INST_V&);
Status AddIdentity(Converter&, const NodeDef&, const T_RTG_INST_V&);
Status AddMaxPool(Converter&, const NodeDef&, const T_RTG_INST_V&);
Status AddRelu(Converter&, const NodeDef&, const T_RTG_INST_V&);
Status AddScale(Converter&, const NodeDef&, const T_RTG_INST_V&);
Status ConvertGraphToRTG(std::unique_ptr<Graph>*, T_INPUT_MAP*);
Status ConvertSubGraphToRTG(std::unique_ptr<Graph>*, Cluster&, T_INPUT_MAP*);
Status RTGToString(Converter&);

} // namspace convert
} // namespace rtglib
} // namespace tensorflow

#endif // TENSORFLOW_RTGLIB_CONVERT_
#endif // TENSORFLOW_USE_ROCM
