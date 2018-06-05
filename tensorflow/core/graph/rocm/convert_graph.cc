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
#include "rocm/include/rtg/operators.hpp"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include <stack>
#include <unordered_map>

#define RTGLIB tensorflow::rtglib

namespace tensorflow {
namespace rtglib {
namespace convert {

Status AddConv2D(Converter& ctx, const NodeDef& nodeDef, const T_RTG_INST_V& inputs) {
    rtg::convolution op;
    string data_format;
    TF_RETURN_IF_ERROR(GetNodeAttr(nodeDef, "data_format", &data_format));
    int h_index = 2;
    int w_index = 3;
    if (ctx.starts_with(data_format, "NHWC")) {
        h_index = 1;
        w_index = 2;
    } else if (ctx.starts_with(data_format, "NCHW")) {
        CHECK(false) << "Unknown data format";
    }
    auto list = nodeDef.attr().at("strides").list();
    std::vector<int> strides;
    strides.push_back(list.i(h_index));
    strides.push_back(list.i(w_index));
    std::copy(strides.begin(), strides.end(), op.stride.begin());
    // TODO: padding, dilations.
    ctx.instructions[nodeDef.name()] = ctx.program->add_instruction(op, inputs);
    return Status::OK();
}

Status AddMaxPool(Converter& ctx, const NodeDef& nodeDef, const T_RTG_INST_V& inputs) {
    CHECK(false);
    return Status::OK();
}

Status AddBiasAdd(Converter& ctx, const NodeDef& nodeDef, const T_RTG_INST_V& inputs) {
    CHECK(false);
    return Status::OK();
}

Status AddConst(Converter& ctx, const NodeDef& nodeDef, const T_RTG_INST_V& inputs) {
    const auto& tensor = nodeDef.attr().at("value").tensor();
    auto& content = tensor.tensor_content();
    DataType dataType;
    ctx.getNodeType(nodeDef, &dataType);
    rtg::shape shape = ctx.getNodeShape(nodeDef, &dataType);
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
    ctx.instructions[nodeDef.name()] = ctx.program->add_literal(li);
    return Status::OK();
}

Status AddIdentity(Converter& ctx, const NodeDef& nodeDef, const T_RTG_INST_V& inputs) {
    CHECK(false);
    return Status::OK();
}

Status AddActivation(Converter& ctx, const NodeDef& nodeDef, const T_RTG_INST_V& inputs) {
    ctx.instructions[nodeDef.name()] = ctx.program->add_instruction(rtg::activation{"relu"}, inputs);
    return Status::OK();
}

Status AddScale(Converter& ctx, const NodeDef& nodeDef, const T_RTG_INST_V& inputs) {
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

void Converter::register_attr_encoders() {
    attr_encoder_registry_["@param"] = EncodeParamAttr;
    attr_encoder_registry_["@literal"] = EncodeConstAttr;
    attr_encoder_registry_["convolution"] = EncodeConvolutionAttr;
    attr_encoder_registry_["activation"] = EncodeActivationAttr;
}

void Converter::register_attr_decoders() {
    attr_decoder_registry_["@param"] = DecodeParamAttr;
    attr_decoder_registry_["@literal"] = DecodeConstAttr;
    attr_decoder_registry_["convolution"] = DecodeConvolutionAttr;
    attr_decoder_registry_["activation"] = DecodeActivationAttr;
}
    
bool Converter::starts_with(const string& value, const string& prefix)
{
    if (prefix.size() <= value.size()) {
        return std::equal(prefix.begin(), prefix.end(), value.begin());
    }
    return false;
}

string Converter::substract_prefix(const string& value, const string& prefix) {
    CHECK(prefix.length() < value.length()) << "unexpected prefix";
    string str = value.substr(prefix.length());
    if (starts_with(str, ":"))
        str = str.substr(1);
#if 0
    if (starts_with(str, Converter::prefix)) {
        str = str.substr(Converter::prefix.length());
        std::size_t pos = str.find(Converter::postfix);
        CHECK(pos != std::string::npos) << "Postfix unfound";
        str = str.substr(pos + Converter::postfix.length());
    }
#endif    
    return str;
}

string Converter::lookupEncoder(const string name)
{
    for (auto iter = attr_encoder_registry_.begin(); iter != attr_encoder_registry_.end(); ++iter) {
        string rtg_name = iter->first;
        if (starts_with(name, rtg_name))
            return rtg_name;
    }
    return "";
}

string Converter::lookupDecoder(const string name)
{
    for (auto iter = attr_decoder_registry_.begin(); iter != attr_decoder_registry_.end(); ++iter) {
        string rtg_name = iter->first;
        if (starts_with(name, rtg_name))
            return rtg_name;
    }
    return "";
}    
    
bool Converter::isRegistered(const Node * node) {
    return op_registry_.count(node->type_string());
}

bool Converter::isCandidate(const Node * node) {
    if (!node->IsOp() || !isRegistered(node))
        return false;
    const char * const gpuDeviceSubStr = "gpu";
    const NodeDef& nodeDef = node->def();
    // If user has specified a non-GPU device.
    if (!nodeDef.device().empty() && !StringPiece(nodeDef.device()).contains(gpuDeviceSubStr))
        return false;
    // If runtime has assigned a non-GPU device.
    if (!node->assigned_device_name().empty() && !StringPiece(node->assigned_device_name()).contains(gpuDeviceSubStr))
        return false;
    return true;
}

void Converter::add_parameter(const NodeDef& nodeDef)  {
    const rtg::shape shape = getNodeShape(nodeDef);
    const string& name = nodeDef.name();
    instructions[name] = program->add_parameter(name, shape);
}

void Converter::add_instruction(const Node* node)  {
    OpConverter op_converter = op_registry_.at(node->type_string());
    T_RTG_INST_V inputs;
    for (const Edge* edge : node->in_edges()) {
        if (edge->IsControlEdge())
            continue;
        const string& name = edge->src()->name();
        CHECK(instructions.find(name) != instructions.end()) << "missing input instruction";
        inputs.push_back(instructions[name]);
    }
    Status s = op_converter(*this, node->def(), inputs);;
    CHECK(s == Status::OK()) << "fail to add instruction";
}

void Converter::decodeAttr(const NameAttrList& func)
{
    string name = func.name();
    string rtg_name = lookupDecoder(name);
    if (rtg_name != "") {
        AttrDecoder attr_decoder = attr_decoder_registry_.at(rtg_name);
        attr_decoder(func, this, rtg_name);
    } else {
        CHECK(false) << "Unknown RTG instruction";
    }
}

DataType Converter::getType(const rtg::shape::type_t& shape_type)
{
    switch (shape_type) {
    case rtg::shape::float_type: return DT_FLOAT; break;
    case rtg::shape::double_type: return DT_DOUBLE; break;
    case rtg::shape::int64_type: return DT_INT64; break;
    case rtg::shape::int32_type: return DT_INT32; break;
    case rtg::shape::int16_type: return DT_INT16; break;
    case rtg::shape::uint16_type: return DT_UINT16; break;
    case rtg::shape::int8_type: return DT_INT8; break;
    default:
        CHECK(false) << "unmatched RTG data type";
    }
}

void Converter::getNodeType(const NodeDef& nodeDef, DataType* data_type)
{
    if (nodeDef.attr().count("dtype")) {
        GetNodeAttr(nodeDef, "dtype", data_type);
    } else if (nodeDef.attr().count("T")) {
        GetNodeAttr(nodeDef, "T", data_type);
    } else {
        CHECK(false) << "data type not found";
    }
}

rtg::shape::type_t Converter::getShapeType(const DataType& data_type)
{
    switch (data_type) {
    case DT_FLOAT: return rtg::shape::float_type; break;
    case DT_DOUBLE: return rtg::shape::double_type; break;
    case DT_INT64: return rtg::shape::int64_type; break;
    // case DT_UINT64: return rtg::shape::uint64_type; break;
    case DT_INT32: return rtg::shape::int32_type; break;
    //    case DT_UINT32: return rtg::shape::uint32_type; break;
    case DT_INT16: return rtg::shape::int16_type; break;
    case DT_UINT16: return rtg::shape::uint16_type; break;
    case DT_INT8: return rtg::shape::int8_type; break;
    default:
        CHECK(false) << "unmatched RTG data type";
    }
}

rtg::shape Converter::getAttrShape(const NameAttrList& func)
{
    auto map = func.attr();
    DataType data_type = map.at("dtype").type();
    const TensorShapeProto & shape_proto = map.at("shape").shape();
    std::vector<std::size_t> dims;
    for (const auto& dim_proto : shape_proto.dim()) {
        int size = dim_proto.size();
        dims.push_back(size);
    }
    rtg::shape::type_t shape_type = getShapeType(data_type);
    return {shape_type, dims};
}

rtg::shape Converter::getNodeShape(const NodeDef& nodeDef, DataType *p_dtype) {
    std::string name = nodeDef.name();
    DataType data_type;
    if (p_dtype == nullptr)
        getNodeType(nodeDef, &data_type);
    else
        data_type = *p_dtype;
    rtg::shape::type_t shape_type = getShapeType(data_type);
    std::vector<std::size_t> dims;
    if (nodeDef.attr().count("value")) {
        const TensorProto& raw_val = nodeDef.attr().at("value").tensor();
        DataType d_type = raw_val.dtype();
        CHECK(data_type == d_type) << "data type unmatched";
        const TensorShape& tensor_shape = raw_val.tensor_shape();
        for (int64 i = 0, e = tensor_shape.dims(); i < e; i++)
            dims.push_back(tensor_shape.dim_size(i));
    } else if (inputs != nullptr) {
        // CHECK(node->type_string() == "_Arg") << "unknown shape";
        CHECK(nodeDef.attr().count("index")) << "unknown argument index";
        int index;
        GetNodeAttr(nodeDef, "index", &index);
        const Tensor tensor = (*inputs)[index].second;
        const TensorShape& tensor_shape = tensor.shape();
        for (int64 i = 0, e = tensor_shape.dims(); i < e; i++)
            dims.push_back(tensor_shape.dim_size(i));
    } else {
        CHECK(false) << "unknown shape";
    }
    
    return {shape_type, dims};
}

void Converter::getTensorShape(const rtg::shape& shape, TensorShape& tensor_shape)
{
    const std::vector<std::size_t>& lens = shape.lens();
    int size = lens.size();
    for (int i = 0; i < size; i++)
        tensor_shape.AddDim(lens[i]);
}

void SetNameAttr(rtg::instruction& ins, NameAttrList& attrs, Converter& convert)
{
    string name = ins.op.name();
    int cnt = 0;
    if (convert.rtgInsCnt.find(name) == convert.rtgInsCnt.end()) {
        convert.rtgInsCnt[name] = 0;
    } else {
        cnt = (convert.rtgInsCnt[name])++;
    }
    string new_name = ins.op.name() + Converter::prefix + std::to_string(cnt) + Converter::postfix;
    attrs.set_name(new_name);
    convert.rtgInsNames[&ins] = new_name;
}
    
void SetInputAttr(rtg::instruction& ins, NameAttrList& attrs, Converter& convert)
{
    auto attr_map = attrs.mutable_attr();
    AttrValue value;
    int32 arg_cnt = ins.arguments.size();
    SetAttrValue(arg_cnt, &value);
    (*attr_map)["num_inputs"] = value;
    arg_cnt = 0;
    for (auto iter = ins.arguments.begin(), end = ins.arguments.end(); iter != end; iter++) {
        rtg::instruction* arg = *iter;
        string name = convert.rtgInsNames[arg];
        AttrValue value;
        SetAttrValue(name, &value);
        string input_name = "input" + std::to_string(arg_cnt);
        arg_cnt++;
        (*attr_map)[input_name] = value;
    }    
}    

void EncodeActivationAttr(rtg::instruction& ins, NameAttrList& attrs, Converter& convert) {
    SetNameAttr(ins, attrs, convert);
    SetInputAttr(ins, attrs, convert);
}
    
void EncodeParamAttr(rtg::instruction& ins, NameAttrList& attrs, Converter& convert) {
    rtg::shape shape = ins.result;
    string name = ins.op.name();
    attrs.set_name(name);
    convert.rtgInsNames[&ins] = name;
    DataType type = convert.getType(shape.type());
    auto attr_map = attrs.mutable_attr();
    AttrValue t_value;
    SetAttrValue(type, &t_value);
    (*attr_map)["dtype"] = t_value;
    TensorShape tensor_shape;
    convert.getTensorShape(shape, tensor_shape);
    AttrValue s_value;
    SetAttrValue(tensor_shape, &s_value);
    (*attr_map)["shape"] = s_value;
}

void EncodeConstAttr(rtg::instruction& ins, NameAttrList& attrs, Converter& convert) {
    SetNameAttr(ins, attrs, convert);
    rtg::shape shape = ins.result;
    DataType type = convert.getType(shape.type());
    auto attr_map = attrs.mutable_attr();
    TensorShape tensor_shape;
    convert.getTensorShape(shape, tensor_shape);
    Tensor tensor(type, tensor_shape);
    int size = tensor.tensor_data().size();
    memcpy(const_cast<char*>(tensor.tensor_data().data()), ins.lit.data(), size);
    TensorProto tensor_proto;
    tensor.AsProtoTensorContent(&tensor_proto);
    AttrValue value;
    SetAttrValue(tensor_proto, &value);
    (*attr_map)["value"] = value;
}

void EncodeConvolutionAttr(rtg::instruction& ins, NameAttrList& attrs, Converter& convert) {
    SetNameAttr(ins, attrs, convert);
    SetInputAttr(ins, attrs, convert);
    // TODO: get stride, padding, dilation
#if 0    
    std::vector<int64> strides;
    rtg::convolution op = ins.op;
    for (auto iter = op.stride.begin(); iter != op.stride.end(); ++iter)
        strides.push_back(*iter);
#endif
    tensorflow::gtl::ArraySlice<int64> value;
}

void DecodeActivationAttr(const NameAttrList& func, Converter* convert, string&prefix) {
    string name = func.name();
    T_RTG_INST_V inputs;
    DecodeInputAttr(inputs, func, convert);
    convert->instructions[name] = convert->program->add_instruction(rtg::activation{"relu"}, inputs);    
}

void DecodeConstAttr(const NameAttrList& func, Converter* convert, string& prefix) {
    string name = func.name();
    auto map = func.attr();
    const auto& tensor = map.at("value").tensor();
    const TensorShapeProto& tensor_shape = tensor.tensor_shape();
    auto& content = tensor.tensor_content();
    DataType data_type = tensor.dtype();
    std::vector<std::size_t> dims;
    for (const auto& dim_proto : tensor_shape.dim()) {
        int size = dim_proto.size();
        dims.push_back(size);
    }
    rtg::shape::type_t shape_type = convert->getShapeType(data_type);
    rtg::shape shape = {shape_type, dims};
    rtg::literal li;
    switch (data_type) {
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
    convert->instructions[name] = convert->program->add_literal(li);
}

void DecodeConvolutionAttr(const NameAttrList& func, Converter* convert, string& prefix) {
    string name = func.name();
    T_RTG_INST_V inputs;
    DecodeInputAttr(inputs, func, convert);
#if 0    
    auto map = func.attr();
    int32 num_of_inputs = map.at("num_inputs").i();
    
    for (int i = 0; i < num_of_inputs; ++i) {
        string input_name = "input" + std::to_string(i);
        string arg_name = map.at(input_name).s();
        CHECK(convert->instructions.find(arg_name) != convert->instructions.end()) << "Input no found";
        inputs.push_back(convert->instructions[arg_name]);
    }
#endif    
    rtg::convolution op;
    // get padding, strides, dilations.
    convert->instructions[name] = convert->program->add_instruction(op, inputs);
}

void DecodeParamAttr(const NameAttrList& func, Converter* convert, string& prefix) {
    string name = func.name();
    const rtg::shape shape = convert->getAttrShape(func);
    string orig_name = convert->substract_prefix(name, prefix);
    convert->instructions[name] = convert->program->add_parameter(orig_name, shape);
}

void DecodeInputAttr(T_RTG_INST_V& inputs, const NameAttrList& func, Converter* convert)
{
    auto map = func.attr();
    int32 num_of_inputs = map.at("num_inputs").i();
    for (int i = 0; i < num_of_inputs; ++i) {
        string input_name = "input" + std::to_string(i);
        string arg_name = map.at(input_name).s();
        CHECK(convert->instructions.find(arg_name) != convert->instructions.end()) << "Input no found";
        inputs.push_back(convert->instructions[arg_name]);
    }
}

Status BuildLaunchNode(std::unique_ptr<Graph>* g, Cluster& cluster, Converter& convert, string& name)
{
    rtg::program* program = convert.program;
    NodeDefBuilder op_builder(name, "RTGLaunchOp");
    std::vector<NodeDefBuilder::NodeOut> income_edges;
    for (const Edge* edge : cluster.input_edges) {
        if (edge->IsControlEdge())
            continue;
        Node* src = edge->src();
        Node* dst = edge->dst();
        int dest_port = edge->dst_input();
        DataType data_type = dst->input_type(dest_port);
        auto income_edge = NodeDefBuilder::NodeOut(src->name(), 
                                                  edge->src_output(), data_type);
        income_edges.emplace_back(income_edge);
    }

    std::vector<DataType> out_types;
    for (const Edge* edge : cluster.output_edges) {
        if (edge->IsControlEdge())
            continue;
        Node* src = edge->src();
        CHECK(!src->IsConstant()) << "todo: constant is exit node";
        Node* dst = edge->dst();
        int dest_port = edge->dst_input();
        DataType data_type = dst->input_type(dest_port);
        out_types.push_back(data_type);
    }

    gtl::ArraySlice<tensorflow::NodeDefBuilder::NodeOut> input_list(income_edges);
    op_builder.Input(input_list);

    unsigned num_values = 0;
    AttrValue value;
    value.mutable_list()->Clear();
    for (auto& ins : program->instructions) {
        num_values++;
        NameAttrList& attrs = *(value.mutable_list()->add_func());
        attrs.Clear();
        string name = ins.op.name();
        string rtg_name = convert.lookupEncoder(name);
        if (rtg_name != "") {
            AttrEncoder attr_encoder = convert.attr_encoder_registry_.at(rtg_name);
            attr_encoder(ins, attrs, convert);
        } else {
            CHECK(false) << "Unknown RTG instruction";
        }
    }

    NameAttrList func;
    func.Clear();
    func.set_name("function");
    (*func.mutable_attr())["func"] = value;
    NodeDef node_def;
    Status status =  op_builder.Attr("function", func)
                        .Attr("OutT", out_types)
                        .Finalize(&node_def);
    CHECK(status.ok()) << "fail to add RTGLaunchOp";
    Graph& graph = **g;
    auto rtg_node = graph.AddNode(node_def, &status);
    TF_RETURN_IF_ERROR(status);

    // Edge info.
    typedef struct {
        Node * src;
        int src_output;
        bool is_control;
    } edge_desc;
    std::vector<edge_desc> input_edges;

    // Cache input edge info.
    for (const Edge* edge : cluster.input_edges)
        input_edges.push_back(edge_desc{edge->src(), edge->src_output(), edge->IsControlEdge()});
    // Construct output edges.
    int ndx = 0;
    for (const Edge* edge : cluster.output_edges) {
        if (!edge->IsControlEdge()) {
            Node* dst = edge->dst();            
            int dest_port = edge->dst_input();    
            TF_RETURN_IF_ERROR(graph.UpdateEdge(rtg_node, ndx, dst, dest_port));
            ndx++;
        }
    }
    // Add control edges at the end.
    for (const Edge* edge : cluster.output_edges) {
        if (edge->IsControlEdge()) {
            graph.RemoveEdge(edge);
            graph.AddControlEdge(rtg_node, edge->dst());
        }
    }
    
    // Remove nodes in the subgraph and their edges.
    for (Node* node : cluster.nodes)
        graph.RemoveNode(node);
    // Construct input edges.
    ndx = 0;
    CHECK(rtg_node->in_edges().empty()) << "Unexpected input edges";
    for (auto iter = input_edges.begin(), end = input_edges.end(); iter != end; iter++) {
        if (!(*iter).is_control) {
            graph.AddEdge((*iter).src, (*iter).src_output, rtg_node, ndx);
            ndx++;
        }
    }
    // Add control edges at the end.
    for (auto iter = input_edges.begin(), end = input_edges.end(); iter != end; iter++) {
        if ((*iter).is_control)
            graph.AddControlEdge((*iter).src, rtg_node);
    }
    rtg_node->set_assigned_device_name(convert.device);

#if 0        
    const TensorShapeProto& shape_proto = def->attr().at("shape").shape();
    for (const auto& dim_proto : shape_proto.dim()) {
        size = dim_proto.size();
    }
    protobuf::TextFormat::PrintToString(graph_def, &serialized);
#endif    
    return Status::OK();    
}

Status ConvertSubgraphToRTG(std::unique_ptr<Graph>* g, Cluster& cluster, T_INPUT_MAP * inputs) {
    rtg::program * program = new rtg::program;
    if (!program)
        return errors::Internal("Fail to create RTG program");

    Converter fwd_convert(program, inputs);
    for (const Edge* edge : cluster.input_edges) {
        if (edge->IsControlEdge())
            continue;
        fwd_convert.add_parameter(edge->src()->def());
    }

    string cluster_name;
    string device;
    for (Node* node : cluster.nodes) {
        fwd_convert.add_instruction(node);
        cluster_name += node->name();
        device = node->assigned_device_name();
    }
    program->print();
    // call program->compile()
    Converter bwd_convert(program, nullptr);
    bwd_convert.device = device;
    TF_RETURN_IF_ERROR(BuildLaunchNode(g, cluster, bwd_convert, cluster_name));

    delete program;
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
        id2Candidate[id] = convert.isCandidate(n) ? true : false;
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

    RTGLIB::dump_graph::DumpGraphToFile("After convert graph to RTG", graph);
    return Status::OK();
}
    
} // namspace convert
} // namespace rtglib
} // namespace tensorflow

#endif // TENSORFLOW_USE_ROCM
