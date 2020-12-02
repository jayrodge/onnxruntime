// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"
#include "orttraining/core/session/training_session.h"

namespace onnxruntime {
namespace training {

void GetPipelineSendOutput(const Graph& graph, std::string& node_arg_name);
void GetPipelineRecvInput(const Graph& graph, std::string& node_arg_name);

Status TransformGraphForPipeline(
    const bool keep_original_output_schema,
    const std::unordered_set<std::string>& weights_to_train,
    const std::unordered_map<std::string, std::vector<int>>& sliced_schema,
    const std::vector<std::string>& expected_output_names,
    Graph& graph,
    pipeline::PipelineTensorNames& pipeline_tensor_names);

Status ApplyPipelinePartitionToMainGraph(
    Graph& graph,
    const std::vector<TrainingSession::TrainingConfiguration::CutInfo>& cut_info,
    size_t pipeline_stage_id,
    size_t num_pipeline_stage);
}  // namespace training
}  // namespace onnxruntime
