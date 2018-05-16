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


#include "tensorflow/core/graph/algorithm.h"
#include "convert_graph.h"
#include "dump_graph.h"
#include <stack>
#include <unordered_map>

#define RTGLIB tensorflow::rtglib

namespace tensorflow {
namespace rtglib {
namespace convert {

Status AddConv2D(Converter& ctx, const Node* node, const T_RTG_SHAPE_V& inputs, T_RTG_SHAPE_V* outputs) {
    return Status::OK();
}

Status AddRelu(Converter& ctx, const Node* node, const T_RTG_SHAPE_V& inputs, T_RTG_SHAPE_V* outputs) {
    CHECK(false);
    return Status::OK();
}

Status AddMaxPool(Converter& ctx, const Node* node, const T_RTG_SHAPE_V& inputs, T_RTG_SHAPE_V* outputs) {
    CHECK(false);
    return Status::OK();
}

Status AddBiasAdd(Converter& ctx, const Node* node, const T_RTG_SHAPE_V& inputs, T_RTG_SHAPE_V* outputs) {
    CHECK(false);
    return Status::OK();
}

Status AddConst(Converter& ctx, const Node* node, const T_RTG_SHAPE_V& inputs, T_RTG_SHAPE_V* outputs) {
    const NodeDef& nodeDef = node->def();
    const auto& tensor = nodeDef.attr().at("value").tensor();
    auto& content = tensor.tensor_content();
    DataType dataType;
    rtg::shape shape = ctx.parse_type(node, dataType);
    rtg::literal li;
    switch (dataType) {
    case DT_FLOAT:{
        const float * ptr = reinterpret_cast<const float*>(content.data());
        int size = content.size()/sizeof(float);
        std::vector<float> data;
        for (int i = 0; i < size; i++)
            data.push_back(ptr[i]);
        li = rtg::literal{shape, data.begin(), data.end()};
        break;
    }
    default:
        CHECK(false) << "unknown data type";
    }
    ctx.instructions[node->name()] = ctx.program->add_literal(li);
    return Status::OK();
}

Status AddIdentity(Converter& ctx, const Node* node, const T_RTG_SHAPE_V& inputs, T_RTG_SHAPE_V* outputs) {
    CHECK(false);
    return Status::OK();
}

Status AddActivation(Converter& ctx, const Node* node, const T_RTG_SHAPE_V& inputs, T_RTG_SHAPE_V* outputs) {
    CHECK(false);
    return Status::OK();
}

Status AddScale(Converter& ctx, const Node* node, const T_RTG_SHAPE_V& inputs, T_RTG_SHAPE_V* outputs) {
    CHECK(false);
    return Status::OK();
}

void Converter::register_op_converters()  {
    op_registry_["Const"] = AddConst;
    op_registry_["Conv2D"] = AddConv2D;
    op_registry_["Relu"] = AddActivation;
#if 0
    op_registry_["BiasAdd"] = AddScale;
    op_registry_["MaxPool"] = AddMaxPool;
    op_registry_["Identity"] = AddIdentity;
#endif    
}

bool Converter::isRegistered(const Node * node) {
    return op_registry_.count(node->type_string());
}

void Converter::add_parameter(const Node* node)  {
    DataType dataType;
    const rtg::shape shape = parse_type(node, dataType);
    const string& name = node->name();
    instructions[name] = program->add_parameter(name, shape);
    shapes[name] = shape;
}

void Converter::add_instruction(const Node* node)  {
    OpConverter op_converter = op_registry_.at(node->type_string());
    T_RTG_SHAPE_V inputs, outputs;
    for (const Edge* edge : node->in_edges()) {
        if (edge->IsControlEdge())
            continue;
        const string& name = node->name();
        CHECK(shapes.find(name) != shapes.end()) << "missing input shape";
        inputs.push_back(shapes[name]);
    }
    Status s = op_converter(*this, node, inputs, &outputs);;
    CHECK(s == Status::OK()) << "fail to add instruction";
}

rtg::shape Converter::parse_type(const Node * node, DataType& data_type) {
    const NodeDef& nodeDef = node->def();
    std::string name = node->name();
    if (nodeDef.attr().count("dtype")) {
        GetNodeAttr(nodeDef, "dtype", &data_type);
    } else if (nodeDef.attr().count("T")) {
        GetNodeAttr(nodeDef, "T", &data_type);
    } else {
        CHECK(false) << "data type not found";
    }
    rtg::shape::type_t shape_type;
    switch (data_type) {
    case DT_FLOAT: shape_type = rtg::shape::float_type; break;
    case DT_DOUBLE: shape_type = rtg::shape::double_type; break;
    case DT_INT64: shape_type = rtg::shape::int64_type; break;
    // case DT_UINT64: shape_type = rtg::shape::uint64_type; break;
    case DT_INT32: shape_type = rtg::shape::int32_type; break;
    //    case DT_UINT32: shape_type = rtg::shape::uint32_type; break;
    case DT_INT16: shape_type = rtg::shape::int16_type; break;
    case DT_UINT16: shape_type = rtg::shape::uint16_type; break;
    case DT_INT8: shape_type = rtg::shape::int8_type; break;
    default:
        CHECK(false) << "unmatched RTG data type";
    }
    
    std::vector<std::size_t> dims;
    if (nodeDef.attr().count("value")) {
        const TensorProto& raw_val = nodeDef.attr().at("value").tensor();
        DataType d_type = raw_val.dtype();
        CHECK(data_type == d_type) << "data type unmatched";
        const TensorShape& tensor_shape = raw_val.tensor_shape();
        for (int64 i = 0, e = tensor_shape.dims(); i < e; i++)
            dims.push_back(tensor_shape.dim_size(i));    
    } else {
        CHECK(node->type_string() == "_Arg") << "unknown shape";
        CHECK(nodeDef.attr().count("index")) << "unknown argument index";
        int index;
        GetNodeAttr(nodeDef, "index", &index);
        const Tensor tensor = (*inputs)[index].second;
        const TensorShape& tensor_shape = tensor.shape();
        for (int64 i = 0, e = tensor_shape.dims(); i < e; i++)
            dims.push_back(tensor_shape.dim_size(i));
    }
    
    return {shape_type, dims};
}

Status ConvertSubgraphToRTG(std::unique_ptr<Graph>* g, Cluster& cluster, T_INPUT_MAP * inputs) {
    rtg::program * program = new rtg::program;
    if (!program)
        return errors::Internal("Fail to create RTG program");

    Converter convert(program, inputs);
    for (const Edge* edge : cluster.input_edges) {
        if (edge->IsControlEdge())
            continue;
        convert.add_parameter(edge->src());
    }

    for (Node* node : cluster.nodes) {
        convert.add_instruction(node);
    }
    program->print();
    return Status::OK();    
}

Status ConvertGraphToRTG(std::unique_ptr<Graph>* g, T_INPUT_MAP* inputs) {
    CHECK_NOTNULL(g);
    const Graph& graph = **g;
    RTGLIB::dump_graph::DumpGraphToFile("Before convert graph to RTG", graph);

    typedef enum {
        is_entry = 0,
        is_exit
    } NodeMask;

    std::unordered_map<int, unsigned> id2Order, id2Segment, id2Mask;
    std::unordered_map<int, bool> id2Candidate, id2Visit;
    std::unordered_map<unsigned, unsigned> segment2Cluster;
    unsigned maxNodeNum = 0;
    unsigned maxSegmentNum = 0;
    unsigned maxClusterNum = 0;
    std::vector<Node *> rpOrder;
    GetReversePostOrder(graph, &rpOrder);
    Converter convert(nullptr, nullptr);

    for (Node* n : rpOrder) {
        int id = n->id();
        id2Order[id] = maxNodeNum++;
        id2Candidate[id] = (n->IsOp() && convert.isRegistered(n)) ? true : false;
        id2Mask[id] = 0;
    }

    Node * sinkNode = graph.sink_node();
    CHECK_NOTNULL(sinkNode);
    std::stack<const tensorflow::Node*> iterStack;
    iterStack.push(sinkNode);
    // iterate graph, mark segments and clusters.
    while (!iterStack.empty()) {
        const Node* node = iterStack.top();
        iterStack.pop();
        int id = node->id();
        if (id2Visit[id])
            continue;

        bool isCandidate = id2Candidate[id];
        // Root of a new segment.
        if (isCandidate && (id2Segment.find(id) == id2Segment.end()))
            id2Segment[id] = maxSegmentNum++;
        id2Visit[id] = true;

        std::unordered_map<int, bool> id2Enqueue;
        for (const Edge* edge : node->in_edges()) {
            bool isCtrlEdge = edge->IsControlEdge();
            Node* nextNode = edge->src();
            int nextId = nextNode->id();
            // Track unique data inputs..            
            if (id2Enqueue.find(nextId) != id2Enqueue.end())
                continue;
            if (id2Order[nextId] >= id2Order[id]) {
                // TODO: Encounter a cycle, might need to detect cycle ahead
                // of time.
                CHECK(false) << "TODO: encounter a circle";
            }
            if (!isCtrlEdge)
                id2Enqueue[nextId] = true;
            bool bothAreCandidates = (isCandidate && id2Candidate[nextId]);
            if (id2Visit.find(nextId) == id2Visit.end()) {
                if (bothAreCandidates && !isCtrlEdge)
                    id2Segment[nextId] = id2Segment[id];
                iterStack.push(nextNode);
            } else if (bothAreCandidates && !isCtrlEdge) {
                // TODO: merge cluster by hashing segment ID to cluster Id.
                CHECK(false) << "TODO: merge segments";
            }

            if (isCandidate && (!id2Candidate[nextId] || isCtrlEdge))
                id2Mask[id] |= (1 << is_entry);
            if ((!isCandidate || isCtrlEdge) && id2Candidate[nextId])
                id2Mask[nextId] |= (1 << is_exit);
        }
    }
    // Assign stand-alone segments to clusters.
    for (unsigned segmentId = 0; segmentId < maxSegmentNum; segmentId++) {
        if (segment2Cluster.find(segmentId) == segment2Cluster.end()) {
            segment2Cluster[segmentId] = maxClusterNum++;
        }
    }

    auto getClusterId = [&] (int nodeId)-> unsigned {
        unsigned segmentId = id2Segment[nodeId];
        unsigned clusterId = segment2Cluster[segmentId];
        return clusterId;
    };

    auto inCluster = [&] (int nodeId)-> bool {
        return  (id2Segment.find(nodeId) != id2Segment.end());
    };
    
    // Build clusters.
    if (maxClusterNum > 0) {
        std::vector<Cluster> clusters;;
        clusters.resize(maxClusterNum);
        for (Node* node : rpOrder) {
            int id = node->id();
            if (!inCluster(id))
                continue;
            unsigned clusterId = getClusterId(id);
            clusters[clusterId].addNode(node);
            if (id2Mask[id] & (1 << is_entry)) {
                for (const Edge* edge : node->in_edges()) {
                    Node* srcNode = edge->src();
                    int srcId = srcNode->id();
                    if (!inCluster(srcId) ||  (getClusterId(srcId) != clusterId)) {
                        clusters[clusterId].addInputEdge(edge);
                    }
                }
            }
            if (id2Mask[id] & (1 << is_exit)) {
                for (const Edge* edge : node->out_edges()) {
                    Node* dstNode = edge->dst();
                    int dstId = dstNode->id();
                    if (!inCluster(dstId) ||  (getClusterId(dstId) != clusterId)) {
                        clusters[clusterId].addOutputEdge(edge);
                    }
                }
            }
        }

        for (unsigned id = 0; id < maxClusterNum; id++) {
            Cluster& cluster = clusters[id];
            if (cluster.getSize() < MIN_CLUSTER_SIZE)
                continue;
            ConvertSubgraphToRTG(g, cluster, inputs);
        }
    }

    return Status::OK();
}
    
} // namspace convert
} // namespace rtglib
} // namespace tensorflow

#endif // TENSORFLOW_USE_ROCM

