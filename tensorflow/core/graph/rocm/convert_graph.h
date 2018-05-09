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
#include "tensorflow/core/framework/node_def_builder.h"

namespace tensorflow {
namespace rtglib {
namespace convert {

#define MIN_CLUSTER_SIZE 3    

class Cluster {
 public:
     explicit Cluster() { init(); }
     void addInputEdge(const Edge* edge)  { input_edges.push_back(edge);  }
     void addOutputEdge(const Edge* edge) { output_edges.push_back(edge); }
     void addNode(Node* node)             { nodes.push_back(node);        }
     int  getSize()                       { return nodes.size();          }
 private:     
     std::vector<const Edge*> input_edges;
     std::vector<const Edge*> output_edges;
     std::vector<Node*> nodes;
     void init() {
         input_edges.clear();
         output_edges.clear();
         nodes.clear();
     }
};    
    
class  Converter;
using OpConverter =
    std::function<tensorflow::Status(Converter&, const tensorflow::NodeDef&)>;
    
class Converter {
public:
    explicit Converter() { Init(); }
    bool IsRegistered(Node* Node);
private:
    std::unordered_map<string, OpConverter> op_registry_;
     void Register_op_converters();
     void Init() {
         Register_op_converters();
     }
};

Status ConvertActivation(Converter& ctx, const NodeDef& node_def); 
Status ConvertBiasAdd(Converter& ctx, const NodeDef& node_def);
Status ConvertConst(Converter& ctx, const NodeDef& node_def); 
Status ConvertConv2D(Converter& ctx, const NodeDef& node_def);
Status ConvertIdentity(Converter& ctx, const NodeDef& node_def);  
Status ConvertMaxPool(Converter& ctx, const NodeDef& node_def);
Status ConvertGraphToRTG(std::unique_ptr<Graph>* graph);
Status ConvertPlaceholder(Converter& ctx, const NodeDef& node_def);
Status ConvertRelu(Converter& ctx, const NodeDef& node_def);
Status ConvertScale(Converter& ctx, const NodeDef& node_def);

} // namspace convert
} // namespace rtglib
} // namespace tensorflow

#endif // TENSORFLOW_RTGLIB_CONVERT_
#endif // TENSORFLOW_USE_ROCM
