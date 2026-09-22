#ifndef A733_NPU_H
#define A733_NPU_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct a733_detection {
    float x1;
    float y1;
    float x2;
    float y2;
    int class_id;
    float confidence;
} a733_detection;

int a733_npu_create(const char *model_path, int class_count, void **detector);
int a733_npu_detect(void *detector, const unsigned char *bgr, int width, int height,
                    float confidence_threshold, a733_detection *detections,
                    int max_detections, int *detection_count);
void a733_npu_destroy(void *detector);
const char *a733_npu_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
