#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

// Repacks a 2D F16 weight that already lives in a Vulkan buffer (ne0 % 128 == 0, ne1 % 64 == 0) into contiguous 64x32 tiles and
// marks it GGML_TENSOR_FLAG_TILED_F16: mul_mat then reads it ~5-35% faster. Only for tensors used as the left operand of a mul_mat
// with more than 8 columns; returns false (tensor untouched) when the tensor does not qualify.
GGML_BACKEND_API bool ggml_backend_vk_tile_f16_weight(struct ggml_tensor * tensor);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

#ifdef  __cplusplus
}
#endif
