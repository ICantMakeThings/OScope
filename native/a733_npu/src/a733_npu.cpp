#include "a733_npu.h"

#include "npulib.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kInputSize = 640;
constexpr int kOutputCount = 6;
constexpr float kNmsThreshold = 0.45f;

struct Detection {
    float x1;
    float y1;
    float x2;
    float y2;
    int class_id;
    float confidence;
};

struct Detector {
    NpuUint npu;
    NetworkItem network;
    int class_count;
    std::vector<unsigned char> input;
};

thread_local std::string last_error;

int fail(const char *message) {
    last_error = message;
    return -1;
}

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

float dfl(const float *values) {
    float maximum = -std::numeric_limits<float>::max();
    for (int index = 0; index < 16; ++index)
        maximum = std::max(maximum, values[index]);
    float total = 0.0f;
    float weighted = 0.0f;
    for (int index = 0; index < 16; ++index) {
        float probability = std::exp(values[index] - maximum);
        total += probability;
        weighted += index * probability;
    }
    return total == 0.0f ? 0.0f : weighted / total;
}

void resize_rgb_bilinear(const unsigned char *source, int source_width, int source_height,
                         unsigned char *target, int target_width, int target_height) {
    for (int y = 0; y < target_height; ++y) {
        float source_y = (y + 0.5f) * source_height / target_height - 0.5f;
        int y0 = std::max(0, std::min(source_height - 1, static_cast<int>(std::floor(source_y))));
        int y1 = std::min(source_height - 1, y0 + 1);
        float fy = source_y - std::floor(source_y);
        for (int x = 0; x < target_width; ++x) {
            float source_x = (x + 0.5f) * source_width / target_width - 0.5f;
            int x0 = std::max(0, std::min(source_width - 1, static_cast<int>(std::floor(source_x))));
            int x1 = std::min(source_width - 1, x0 + 1);
            float fx = source_x - std::floor(source_x);
            const unsigned char *p00 = source + (y0 * source_width + x0) * 3;
            const unsigned char *p01 = source + (y0 * source_width + x1) * 3;
            const unsigned char *p10 = source + (y1 * source_width + x0) * 3;
            const unsigned char *p11 = source + (y1 * source_width + x1) * 3;
            unsigned char *out = target + (y * target_width + x) * 3;
            for (int channel = 0; channel < 3; ++channel) {
                float top = p00[channel] + (p01[channel] - p00[channel]) * fx;
                float bottom = p10[channel] + (p11[channel] - p10[channel]) * fx;
                out[channel] = static_cast<unsigned char>(std::round(top + (bottom - top) * fy));
            }
        }
    }
}

void letterbox_rgb(const unsigned char *bgr, int width, int height, std::vector<unsigned char> &output) {
    float scale = std::min(kInputSize / static_cast<float>(width), kInputSize / static_cast<float>(height));
    int resized_width = static_cast<int>(std::round(width * scale));
    int resized_height = static_cast<int>(std::round(height * scale));
    int pad_x = (kInputSize - resized_width) / 2;
    int pad_y = (kInputSize - resized_height) / 2;
    std::vector<unsigned char> resized(resized_width * resized_height * 3);
    resize_rgb_bilinear(bgr, width, height, resized.data(), resized_width, resized_height);
    output.assign(kInputSize * kInputSize * 3, 114);
    for (int y = 0; y < resized_height; ++y) {
        for (int x = 0; x < resized_width; ++x) {
            const unsigned char *pixel = resized.data() + (y * resized_width + x) * 3;
            unsigned char *destination = output.data() + ((y + pad_y) * kInputSize + x + pad_x) * 3;
            destination[0] = pixel[2];
            destination[1] = pixel[1];
            destination[2] = pixel[0];
        }
    }
}

float intersection_over_union(const Detection &left, const Detection &right) {
    float x1 = std::max(left.x1, right.x1);
    float y1 = std::max(left.y1, right.y1);
    float x2 = std::min(left.x2, right.x2);
    float y2 = std::min(left.y2, right.y2);
    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float left_area = std::max(0.0f, left.x2 - left.x1) * std::max(0.0f, left.y2 - left.y1);
    float right_area = std::max(0.0f, right.x2 - right.x1) * std::max(0.0f, right.y2 - right.y1);
    return intersection / std::max(0.000001f, left_area + right_area - intersection);
}

void decode(const float *const *outputs, int class_count, int width, int height,
            float threshold, std::vector<Detection> &decoded) {
    float scale = std::min(kInputSize / static_cast<float>(width), kInputSize / static_cast<float>(height));
    int resized_width = static_cast<int>(std::round(width * scale));
    int resized_height = static_cast<int>(std::round(height * scale));
    float pad_x = (kInputSize - resized_width) / 2.0f;
    float pad_y = (kInputSize - resized_height) / 2.0f;
    const int strides[] = {8, 16, 32};
    for (int level = 0; level < 3; ++level) {
        int grid_width = kInputSize / strides[level];
        int grid_height = grid_width;
        int grid_size = grid_width * grid_height;
        const float *box_output = outputs[level * 2];
        const float *score_output = outputs[level * 2 + 1];
        for (int y = 0; y < grid_height; ++y) {
            for (int x = 0; x < grid_width; ++x) {
                int grid_index = y * grid_width + x;
                int class_id = 0;
                float best_score = -std::numeric_limits<float>::max();
                for (int class_index = 0; class_index < class_count; ++class_index) {
                    float score = score_output[class_index * grid_size + grid_index];
                    if (score > best_score) {
                        best_score = score;
                        class_id = class_index;
                    }
                }
                float confidence = sigmoid(best_score);
                if (confidence < threshold)
                    continue;
                const float *box = box_output + grid_index * 64;
                float x1 = (x + 0.5f - dfl(box)) * strides[level];
                float y1 = (y + 0.5f - dfl(box + 16)) * strides[level];
                float x2 = (x + 0.5f + dfl(box + 32)) * strides[level];
                float y2 = (y + 0.5f + dfl(box + 48)) * strides[level];
                Detection item;
                item.x1 = std::max(0.0f, std::min(width - 1.0f, (x1 - pad_x) / scale));
                item.y1 = std::max(0.0f, std::min(height - 1.0f, (y1 - pad_y) / scale));
                item.x2 = std::max(0.0f, std::min(width - 1.0f, (x2 - pad_x) / scale));
                item.y2 = std::max(0.0f, std::min(height - 1.0f, (y2 - pad_y) / scale));
                item.class_id = class_id;
                item.confidence = confidence;
                decoded.push_back(item);
            }
        }
    }
}

void nms(std::vector<Detection> &detections) {
    std::sort(detections.begin(), detections.end(), [](const Detection &left, const Detection &right) {
        return left.confidence > right.confidence;
    });
    std::vector<Detection> kept;
    for (const Detection &candidate : detections) {
        bool suppressed = false;
        for (const Detection &existing : kept) {
            if (intersection_over_union(candidate, existing) > kNmsThreshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed)
            kept.push_back(candidate);
    }
    detections.swap(kept);
}

}  // namespace

extern "C" int a733_npu_create(const char *model_path, int class_count, void **detector) {
    if (!model_path || !detector || class_count <= 0)
        return fail("invalid A733 detector arguments");
    Detector *instance = new Detector();
    instance->class_count = class_count;
    if (instance->npu.npu_init() != 0) {
        delete instance;
        return fail("A733 NPU initialization failed");
    }
    unsigned int network_id = 0;
    if (instance->network.network_create(const_cast<char *>(model_path), network_id) != 0 ||
        instance->network.network_prepare() != 0 || instance->network.get_output_cnt() != kOutputCount) {
        delete instance;
        return fail("A733 model loading failed; verify the .nb model and runtime libraries");
    }
    void *input = nullptr;
    unsigned int input_size = 0;
    if (instance->network.get_network_input_buff_info(0, &input, &input_size) != 0 ||
        input_size < kInputSize * kInputSize * 3) {
        delete instance;
        return fail("A733 model input is not a 640x640 RGB tensor");
    }
    instance->input.resize(kInputSize * kInputSize * 3);
    *detector = instance;
    last_error.clear();
    return 0;
}

extern "C" int a733_npu_detect(void *opaque, const unsigned char *bgr, int width, int height,
                                float confidence_threshold, a733_detection *detections,
                                int max_detections, int *detection_count) {
    if (!opaque || !bgr || width <= 0 || height <= 0 || !detections || max_detections < 0 || !detection_count)
        return fail("invalid A733 detection arguments");
    Detector *instance = static_cast<Detector *>(opaque);
    letterbox_rgb(bgr, width, height, instance->input);
    if (instance->network.network_load_input_buffer(instance->input.data(), instance->input.size()) != 0 ||
        instance->network.network_input_output_set() != 0 || instance->network.network_run() != 0) {
        return fail("A733 NPU inference failed");
    }
    std::vector<float *> output_storage(kOutputCount, nullptr);
    if (!instance->network.get_output(output_storage.data()))
        return fail("A733 output retrieval failed");
    std::vector<Detection> decoded;
    decode(output_storage.data(), instance->class_count, width, height,
           std::max(0.0f, confidence_threshold), decoded);
    nms(decoded);
    int count = std::min(max_detections, static_cast<int>(decoded.size()));
    for (int index = 0; index < count; ++index) {
        detections[index].x1 = decoded[index].x1;
        detections[index].y1 = decoded[index].y1;
        detections[index].x2 = decoded[index].x2;
        detections[index].y2 = decoded[index].y2;
        detections[index].class_id = decoded[index].class_id;
        detections[index].confidence = decoded[index].confidence;
    }
    *detection_count = count;
    for (float *output : output_storage)
        std::free(output);
    last_error.clear();
    return 0;
}

extern "C" void a733_npu_destroy(void *opaque) {
    delete static_cast<Detector *>(opaque);
}

extern "C" const char *a733_npu_last_error(void) {
    return last_error.c_str();
}
