#include <algorithm>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {

constexpr unsigned int kValuesPerPrediction = 6;

float clip(float value, float lower, float upper) {
    return std::max(lower, std::min(value, upper));
}

}  // namespace

extern "C" bool NvDsInferParseCustomYoloV5(
    const std::vector<NvDsInferLayerInfo>& output_layers,
    const NvDsInferNetworkInfo& network_info,
    const NvDsInferParseDetectionParams& detection_params,
    std::vector<NvDsInferObjectDetectionInfo>& objects) {
    if (output_layers.empty() || output_layers[0].buffer == nullptr) {
        std::cerr << "YOLOv5 parser: output0 is missing" << std::endl;
        return false;
    }

    const NvDsInferLayerInfo& output = output_layers[0];
    if (output.dataType != FLOAT) {
        std::cerr << "YOLOv5 parser: output0 must be FP32" << std::endl;
        return false;
    }
    if (output.inferDims.numElements % kValuesPerPrediction != 0) {
        std::cerr << "YOLOv5 parser: unexpected output element count "
                  << output.inferDims.numElements << std::endl;
        return false;
    }

    const float threshold = detection_params.perClassPreclusterThreshold.empty()
        ? 0.25F
        : detection_params.perClassPreclusterThreshold[0];
    const float* predictions = static_cast<const float*>(output.buffer);
    const unsigned int prediction_count =
        output.inferDims.numElements / kValuesPerPrediction;
    const float max_x = static_cast<float>(network_info.width - 1);
    const float max_y = static_cast<float>(network_info.height - 1);

    objects.reserve(objects.size() + prediction_count / 100);
    for (unsigned int index = 0; index < prediction_count; ++index) {
        const float* row = predictions + index * kValuesPerPrediction;
        const float confidence = row[4] * row[5];
        if (confidence < threshold) {
            continue;
        }

        const float left = clip(row[0] - row[2] * 0.5F, 0.0F, max_x);
        const float top = clip(row[1] - row[3] * 0.5F, 0.0F, max_y);
        const float right = clip(row[0] + row[2] * 0.5F, 0.0F, max_x);
        const float bottom = clip(row[1] + row[3] * 0.5F, 0.0F, max_y);
        if (right <= left || bottom <= top) {
            continue;
        }

        NvDsInferObjectDetectionInfo object{};
        object.classId = 0;
        object.left = left;
        object.top = top;
        object.width = right - left;
        object.height = bottom - top;
        object.detectionConfidence = confidence;
        objects.push_back(object);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV5);
