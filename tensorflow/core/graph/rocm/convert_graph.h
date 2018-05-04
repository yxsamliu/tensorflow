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

class Segment {
 public:
     explicit Segment() { init(); }
 private:
     std::vector<Node*> nodes;
     void init() { nodes.clear(); }
 };    

    
class  Converter;
using OpConverter =
    std::function<tensorflow::Status(Converter&, const tensorflow::NodeDef&)>;
    
class Converter {
public:
    explicit Converter() { Init(); }
    bool IsSegmentCandidate(Node* node);
    bool IsRegistered(Node* Node);
    bool IsLeaf(Node * node);
    enum SegmentNodeAttr{
        Visited = 0,
        IsExit,
        IsEntry,
        IsInput,
        IsCandidate
    };
 
private:
    std::unordered_map<string, OpConverter> op_registry_;
    std::unordered_map<int, int> segmentMap;
     std::unordered_map<int, int64> segmentNodeAttrMap;
     int maxSegmentId;
     void Register_op_converters();
     void Init() {
         Register_op_converters();
         maxSegmentId = 0;
         segmentMap.clear();
         segmentNodeAttrMap.clear();
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
