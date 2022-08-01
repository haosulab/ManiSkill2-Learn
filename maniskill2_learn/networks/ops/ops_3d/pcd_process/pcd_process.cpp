/* Modified from [Use double in some computation to improve numerical stability
 * https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query.cpp
 */

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

// at::Tensor VoxelDownSample(const at::Tensor &xyz, const float &z_min, const int &num);

// at::Tensor ManiSkillDownSample(const at::Tensor &xyz, const float &min_z, const int &num);

std::tuple<at::Tensor, at::Tensor> UniformDownSample(const at::Tensor &xyz, const float &min_z, const int &num);

std::tuple<at::Tensor, at::Tensor> ManiSkillDownSample(const at::Tensor &xyz, const at::Tensor &seg, const float &min_z, const int &num, const int &num_min,
                                                       const int &num_fg);

// at::Tensor cumsum(const at::Tensor &mask);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("voxel_downsample", &VoxelDownsample, "VoxelDownsample");
    m.def("uniform_downsample", &UniformDownSample, "UniformDownSample", py::arg("xyz"), py::arg("min_z"), py::arg("num"));
    m.def("maniskill_downsample", &ManiSkillDownSample, "ManiSkillDownSample", py::arg("xyz"), py::arg("seg"), py::arg("min_z"), py::arg("num"),
          py::arg("num_min"), py::arg("num_fg"));
}
