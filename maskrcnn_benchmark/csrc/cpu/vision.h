// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/extension.h>


at::Tensor ROIAlign_forward_cpu(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width,
                                const int sampling_ratio);


at::Tensor nms_cpu(const at::Tensor& dets,
                   const at::Tensor& scores,
                   const float threshold);

std::tuple<at::Tensor, at::Tensor> soft_nms_cpu(at::Tensor& dets,
               at::Tensor& scores,
               const float nms_thresh,
               const float sigma,
               const float score_thresh,
               const int method);