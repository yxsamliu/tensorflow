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
    op_registry_["Const"] = ConvertConst;
    op_registry_["Conv2D"] = ConvertConv2D;
    op_registry_["BiasAdd"] = ConvertScale;
    op_registry_["Relu"] = ConvertActivation;
#if 0    
    op_registry_["MaxPool"] = ConvertMaxPool;
    op_registry_["Identity"] = ConvertIdentity;
#endif    
}

bool Converter::IsRegistered(Node * node) {
    return op_registry_.count(node->type_string());
}

Status ConvertGraphToRTG(std::unique_ptr<Graph>* g) {
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
    Converter convert;

    for (Node* n : rpOrder) {
        int id = n->id();
        id2Order[id] = maxNodeNum++;
        id2Candidate[id] = (n->IsOp() && convert.IsRegistered(n)) ? true : false;
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
                CHECK(false);
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
                CHECK(false);
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
        }
    }

    return Status::OK();
}
    
} // namspace convert
} // namespace rtglib
} // namespace tensorflow

#endif // TENSORFLOW_USE_ROCM

