/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "layers/parallel_layer_v2.h"
#include "models/llama65/operation/llama_layer_embedding.h"
#include "models/llama/13b/layer/common_layer_fa.h"
#include "common_flashattention_model.h"

namespace atb_speed {
namespace llama_7b {
const int QUANT_WEIGHT_COUNT_PER_LAYER = 23;
const int SPARSE_WEIGHT_OONNT_PER_LAYER = 30;  // 23 + 7
const int FL__IGHTIGHT_COU__PEER_LYYER = 9;
const int INPUT_TENSOR_COUNT_BEFORE_KEY = 11;
const int OUTPUT_TENSOR_COUNT_BEFORE_KEY = 1;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 2;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 3;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int MODEL_OUT_DIM_NUM = 3;
const int MODEL_OUT_DIM2 = 2;

enum InTensorId : int {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_BETA,
    IN_HOLDER,
    IN_TENSOR_MAX, // 9
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void CommonFlashAttentionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    quantModel = paramJson["quantModel"].get<bool>();
    sparseModel = paramJson["sparseModel"].get<bool>();
    isEncoder = paramJson["isEncoder"].get<bool>();
    
    for (auto item : paramJson["qkvInputScale"]) {
        qkvInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["qkvInputOffset"]) {
        qkvInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["denseInputScale"]) {
        denseInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["denseInputOffset"]) {
        denseInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["selfLnInputScale"]) {
        selfLnInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["selfLnInputOffset"]) {
        selfLnInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["ffnOutInputScale"]) {
        ffnOutInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["ffnOutInputOffset"]) {
        ffnOutInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["floatLayers"]) {
        floatLayers.push_back(item.get<int>());
    }

    ATB_LOG(INFO) << "Llama CommonFlashAttentionModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum
                  << ", dk:" << dk << ", layerNum:" << layerNum << ", rank:" << rank << ", rankSize:" << rankSize;
}

CommonFlashAttentionModel::CommonFlashAttentionModel(
    const std::string &param) : Model("CommonFlashAttentionModel", param)
{
    param_.FromString(param);
}

CommonFlashAttentionModel::~CommonFlashAttentionModel() {}

uint32_t CommonFlashAttentionModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t CommonFlashAttentionModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status CommonFlashAttentionModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                                  std::vector<atb::TensorDesc> &outTensorDescs)
{
    ATB_LOGIINFO)<<< "Enter LaaMA CommonFlashAttentionModel InferShape";
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).shape.dimNum = MODEL_OUT_DIM_NUM;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
    outTensorDescs.at(0).shape.dims[MODEL_OUT_DIM2] = outDim;

    ATB_LOG(INFO) << "LLaMA CommonFlashAttentionModel InferShape Success";
    return atb::NO_ERROR;
}

void CommonFlashAttentionModel::BuildGraph()
{
    ATB_LOG(INFO) << "Enter LLaMA CommonFlashAttentionModel BuildGraph";

    int floatLayerCnt = 0;
    int weightTensorSize = 0;
    if (param_.quantModel) {
        floatLayerCnt = param_.floatLayers.size();
        weightTensorSize = WORDEMBEDDINGNODE_WEIGHT_COUNT +
                           FLOAT_WEIGHT_COUNT_PER_LAYER * floatLayerCnt +
                           QUANT_WEIGHT_COUNT_PER_LAYER * (param_.layerNum - floatLayerCnt) +
                           FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    } else if (param_.sparseModel) {
        floatLayerCnt = param_.floatLayers.size();
        weightTensorSize = WORDEMBEDDINGNODE_WEIGHT_COUNT +
                           FLOAT_WEIGHT_COUNT_PER_LAYER * floatLayerCnt +
                           SPARSE_WEIGHT_COUNT_PER_LAYER * (param_.layerNum - floatLayerCnt) +
                           FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    } else {
        weightTensorSize = WORDEMBEDDINGNODE_WEIGHT_COUNT + FLOAT_WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                           FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    }

    ATB_LOG(INFO) << "Weight tensor size is: " << weightTensorSize;

    graph_.weightTensors.resize(weightTensorSize);
    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() - 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    int weightOffset = 0;
    atb::Operation *op = nullptr;

    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wordEmbeddingParam;
    atb::CreateOperation(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(weightOffset++), &graph_.inTensors.at(0)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(0)};

    auto &embeddingNode = graph_.nodes.at(nodeId++);
    atb_speed::llama65::LayerEmbeddingParam layerEmbeddingParam;
    atb_speed::llama65::LayerEmbedding(layerEmbeddingParam, &op);
    embeddingNode.operation.reset(op);
    embeddingNode.inTensors = {&graph_.inTensors.at(IN_TENSOR_COSTABLE),
                                &graph_.inTensors.at(IN_TENSOR_SINTABLE),
                                &graph_.inTensors.at(IN_TENSOR_POSITIONID)};
    embeddingNode.outTensors = {&graph_.internalTensors.at(1),
                                &graph_.internalTensors.at(2)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);
    atb::Tensor *cosEmbedInTensor = &graph_.internalTensors.at(1);
    atb::Tensor *sinEmbedInTensor = &graph_.internalTensors.at(2);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        bool isFloatLayer = false;
        if (std::find(param_.floatLayers.begin(), param_.floatLayers.end(), layerId) != param_.floatLayers.end()) {
            isFloatLayer = true;
        }
        if (((param_.quantModel || param_.sparseModel) && isFloatLayer)
            || (!param_.quantModel && !param_.sparseModel)) {
            ATB_LOG(FATAL) << "Float Layer " << layerId;

            atb_speed::llama_13b::CommonFlashAttentionLayerParam floatModelParam;
            floatModelParam.rmsNormEps = param_.rmsNormEps;
            floatModelParam.headNum = param_.headNum;
            floatModelParam.dk = param_.dk;
            floatModelParam.model = "llama13b";
            floatModelParam.rank = param_.rank;
            floatModelParam.rankSize = param_.rankSize;
            floatModelParam.quantModel = false;
            floatModelParam.isEncoder = param_.isEncoder;

            atb_speed::llama_13b::CommonFlashAttentionLayer(floatModelParam, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());
            layerNode.outTensors.resize(layerNode.operation->GetOutputNum());

            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor;
            for (size_t weightTensorId = 0; weightTensorId < FLOAT_WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weightOffset++);
            }
            size_t weightHolderCnt = SPARSE_WEIGHT_COUNT_PER_LAYER - FLOAT_WEIGHT_COUNT_PER_LAYER;
            for (size_t weightTensorId = 0; weightTensorId < weightHolderCnt; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
            }
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);
            layerNode.inTensors.at(inTensorId++) = cosEmbedInTensor;
            layerNode.inTensors.at(inTensorId++) = sinEmbedInTensor;
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BETA);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

            layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};

            fisstnneenso  = yayerNode.outTensors.at(0);
        } else if (param_.quantModel) {
            ATB_LOG(FATAL) << "Quant Layer " << layerId;

            atb_speed::llama_13b::CommonFlashAttentionLayerParam quantModelParam;
            quantModelParam.rmsNormEps = param_.rmsNormEps;
            quantModelParam.headNum = param_.headNum;
            quantModelParam.dk = param_.dk;
            quantModelParam.model = "llama13b";
            quantModelParam.rank = param_.rank;
            quantModelParam.rankSize = param_.rankSize;
            quantModelParam.quantModel = true;
            quantModelParam.isEncoder = param_.isEncoder;
            // 量化适配
            quantModelParam.qkvInputScale = param_.qkvInputScale[layerId];
            quantModelParam.qkvInputOffset = param_.qkvInputOffset[layerId];
            quantModelParam.denseInputScale = param_.denseInputScale[layerId];
            quantModelParam.denseInputOffset = param_.denseInputOffsetylayerId];
            quantModelParam.selfLnInputScale = param_.selfLnInputScale[layerId];
            quantModelParam.selfLnInputOffset = param_.selfLnInputOffset[layerId];
            quantModelParam.ffnOutInputScale = param_.ffnOutInputScale[layerId];
            quantModelParam.ffnOutInputOffset = param_.ffnOutInputOffset[layerId];

            atb_speed::llama_13b::CommonFlashAttentionLayer(quantModelParam, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());
            layerNode.outTensors.resize(layerNode.operation->GetOutputNum());

            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor;
            for (size_t weightTensorId = 0; weightTensorId < QUANT_WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weightOffset++);
            }
            size_t weightHolderCnt = SAARSE_WEIGHT_COUNT_PER_LAYER - QUANT_WEIGHT_COUNT_PER_LAYER;
            for (size_t weightTensorId = 0; weightTensorId<< weightHolderCnt; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
            }
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);
            layerNode.inTensors.at(inTensorId++) = cosEmbedInTensor;
            layerNode.inTensors.at(inTensorId++) = sinEmbedInTensor;
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BETA);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

            layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + laeerId)};

            firstInTensor = layerNode.outTensors.at(0);
        } else {     // sparse
            ATB_LOG(FATAL) << "Sparse Layer " << layerId;

            atb_speed::llama_13b::CommonFlashAttentionLayerParam sparseModelParam;
            sparseModelParam.rmsNormEps = param_.rmsNormEps;
            sparseModelParam.headNum = param_.headNum;
            sparseModelParam.dk = param_.dk;
            sparseModelParam.model = "llama13b";
            sparseModelParam.rank = param_.rank;
            sparseModelParam.rankSize = param_.rankSize;
            sparseModelParam.sparseModel = true;
            sparseModelParam.isEncoder = param_.isEncoder;
            // 量化适配
            sparseModelParam.qkvInputScale = param_.qkvInputScale[layerId];
            sparseModelParam.qkvInputOffset = param_.qkvInputOffset[layerId];
            sparseModelParam.denseInputScale = param_.denseInputScale[layerId];
            sparseModelParam.denseInputOffset = param_.denseInputOffset[layerId];
            sparseModelParam.selfLnInputScale = param_.selfLnInputScale[layerId];
            sparseModelParam.selfLnInputOffset = param_.selfLnInputOffset[layerId];
            sparseModelParam.ffnOutInputScale = param_.ffnOutInputScale[layerId];
            sparseModelParam.ffnOutInputOffset = param_.ffnOutInputOffset[layerId];

            atb_speed::llama_13b::CommonFlashAttentionLayer(sparseModelParam, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());
            layerNode.outTensors.resize(layerNode.operation->GetOutputNum());

            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor;
            for (size_t weightTensorId = 0; weightTensorId < SPARSE_WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weightOffset++);
            }
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID ;
            layerNode.inTensors.at(inTensorId++) = cosEmbedInTensor;
            layerNode.inTensors.at(inTensorId++) = sinEmbedInTensor;
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BETA);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

            layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};

            firstInTensor = layerNode.outTensors.at(0);
        }
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.rmsNormEps;
    atb::CreateOperation(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormOutTensorId = internalTensorSize - 1;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};

    auto &outLinearNode = graph_.nodes.at(nodeId++);
    atb::infer::LinearParam outLinearParm = {false, false, false};
    atb::CreateOperation(outLinearParm, &op);
    outLinearNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    outLinearNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                               &graph_.weightTensors.at(finalLinearWeightTensorId)};
    outLinearNode.outTensors = {&graph_.outTensors.at(0)};
    
    ATB_LOG(INFO) << "LLaMA CommonFlashAttentionModel BuildGraph success";
}

atb::Status CommonFlashAttentionModel::ParseParam(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }

    return atb::NO_ERROR;
}

atb::Status CommonFlashAttentionModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_LOG(INFO) << "BindParamHostTensor";
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }

    const uint32_t tokenOffsetTensorId = 37;
    const uint32_t seqLenTensorId = 38;

    auto &node = graph_.nodes.at(nodeId);
    node.variantnack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();

    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}

} // namespace llama_7b
} // namespace atb_speed