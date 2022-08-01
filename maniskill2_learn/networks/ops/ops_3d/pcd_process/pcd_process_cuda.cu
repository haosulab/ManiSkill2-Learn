#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <torch/extension.h>
using namespace torch::indexing;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x)                                                                                                                                         \
    CHECK_CUDA(x);                                                                                                                                             \
    CHECK_CONTIGUOUS(x)

#define MAX_NUM_SEGS 5
#define THREADS_PER_BLOCK 1024
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void uniform_downsample_kernel(const float *__restrict__ xyz, bool *__restrict__ mask, int64_t *__restrict__ index, int64_t *__restrict__ rand_idx,
                                          const int B, const int N, const int num, const float min_z) {
    __shared__ int count, valid_num;
    const int batch_index = blockIdx.x;
    int idx;

    if (batch_index >= B)
        return;

    xyz += batch_index * N * 3, mask += batch_index * num, index += batch_index * num;

    if (threadIdx.x == 0)
        count = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < N; i += THREADS_PER_BLOCK) {
        if (xyz[rand_idx[i] * 3 + 2] <= min_z)
            continue;
        idx = atomicAdd(&count, 1);
        if (idx >= num)
            break;
        index[idx] = rand_idx[i], mask[idx] = true;
    }

    // Padding num
    __syncthreads();
    if (threadIdx.x == 0) {
        valid_num = count;
        if (count == 0)
            count = idx = valid_num = 1;
    }
    __syncthreads();

    while (idx < num) {
        idx = atomicAdd(&count, 1);
        if (idx >= num)
            break;
        index[idx] = index[(idx - valid_num) % valid_num];
    }
}

std::tuple<at::Tensor, at::Tensor> UniformDownSample(const at::Tensor &xyz, const float &min_z, const int &num) {
    CHECK_INPUT(xyz);
    TORCH_CHECK(xyz.dim() == 3 and xyz.size(2) == 3, "Pointcloud must have shape (B, N, 3)");

    at::cuda::CUDAGuard device_guard(xyz.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int B = xyz.size(0), N = xyz.size(1);
    auto xyz_options = xyz.options();
    auto int64_options = xyz_options.dtype(torch::kInt64), bool_options = xyz_options.dtype(torch::kBool);

    auto mask = at::zeros({B, num}, bool_options);
    auto index = at::zeros({B, num}, int64_options);
    auto rand_idx = torch::randperm(N, int64_options);

    dim3 blocks(B), threads(THREADS_PER_BLOCK);
    uniform_downsample_kernel<<<blocks, threads, 0, stream>>>(xyz.data_ptr<float>(), mask.data_ptr<bool>(), index.data_ptr<int64_t>(),
                                                              rand_idx.data_ptr<int64_t>(), B, N, num, min_z);

    AT_CUDA_CHECK(cudaGetLastError());

    return std::make_tuple(index, mask);
}

__global__ void maniskill_downsample_kernel(const int64_t *__restrict__ num_pts, const bool *__restrict__ seg, bool *__restrict__ mask,
                                            int64_t *__restrict__ index, int64_t *__restrict__ rand_idx, const int B, const int N, const int K, const int num,
                                            const float min_z) {

    // xyz.data_ptr<float>(), all_pts.data_ptr<int64_t>(), all_sign.data_ptr<bool>(),
    // mask.data_ptr<bool>(), index.data_ptr<int64_t>(), rand_idx.data_ptr<int64_t>(),
    // B, N, K + 1, num, min_z);

    // xyz.data_ptr<float>(), all_pts.data_ptr<bool>(), all_sign.data_ptr<bool>(),
    // mask.data_ptr<bool>(), index.data_ptr<int64_t>(), rand_idx.data_ptr<int64_t>(),
    // B, N, K, num, min_z);
    __shared__ int count, valid_num, cnt_segs[MAX_NUM_SEGS];
    const int batch_index = blockIdx.x;
    int idx, seg_idx, cnt;

    if (batch_index >= B)
        return;

    num_pts += batch_index * K, seg += batch_index * N * K;
    mask += batch_index * num, index += batch_index * num;

    if (threadIdx.x == 0) {
        memset(cnt_segs, 0, sizeof(cnt_segs));
        count = 0;
    }
    __syncthreads();

    int i;
    for (i = threadIdx.x; i < N; i += THREADS_PER_BLOCK) {
        seg_idx = -1;
        for (int j = 0; j < K; j++)
            if (seg[rand_idx[i] * K + j]) {
                seg_idx = j;
                break;
            }
        if (seg_idx < 0)
            continue;

        if (cnt_segs[seg_idx] >= num_pts[seg_idx] or atomicAdd(cnt_segs + seg_idx, 1) >= num_pts[seg_idx])
            continue;

        idx = atomicAdd(&count, 1);
        if (idx >= num)
            break;

        // printf("CUDA %d %ld %d\n", idx, rand_idx[i], i);
        index[idx] = rand_idx[i], mask[idx] = true;
    }
    // Padding num
    __syncthreads();
    if (threadIdx.x == 0) {
        valid_num = count;
        if (count == 0)
            count = idx = valid_num = 1;
    }
    __syncthreads();
    /*
        if (threadIdx.x == 0) {
        printf("CUDA: B %d, th 0 last idx %d, total %d, final idx: %d/%d\n", batch_index, idx, count, i, N);
        for (int j = 0; j < K; j++)
            printf("CUDA SEG %d %d %d %ld\n", batch_index, j, cnt_segs[j], num_pts[j]);
    }*/

    while (idx < num) {
        idx = atomicAdd(&count, 1);
        if (idx >= num)
            break;
        index[idx] = index[idx % valid_num];
    }
}

std::tuple<at::Tensor, at::Tensor> ManiSkillDownSample(const at::Tensor &xyz, const at::Tensor &seg, const float &min_z, const int &num, const int &num_min,
                                                       const int &num_fg) {
    at::TensorArg xyz_t{xyz, "xyz", 1}, seg_t{seg, "seg", 2};
    at::CheckedFrom c = "ManiSkillDownSample";
    at::checkAllSameGPU(c, {xyz_t, seg_t});

    CHECK_INPUT(xyz);
    CHECK_INPUT(seg);
    TORCH_CHECK(xyz.dim() == 3 and xyz.size(2) == 3, "Pointcloud must have shape (B, N, 3).");
    TORCH_CHECK(seg.dim() == 3 and xyz.size(2) < MAX_NUM_SEGS, "Segmentation must have shape (B, N, K) and, K needs to be less than MAX_NUM_SEGS.");

    at::cuda::CUDAGuard device_guard(xyz.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int B = xyz.size(0), N = xyz.size(1);
    auto xyz_options = xyz.options();
    auto int64_options = xyz_options.dtype(torch::kInt64), bool_options = xyz_options.dtype(torch::kBool);

    auto mask = at::zeros({B, num}, bool_options);
    auto index = at::zeros({B, num}, int64_options);
    auto rand_idx = torch::randperm(N, int64_options);

    auto z_mask = xyz.index({"...", Slice(2)}) > min_z;                                             // [B, N, 1]
    auto fg = seg.any(-1, true), bg = at::logical_and(at::logical_not(fg), z_mask);                 // [B, N, 1], [B, N, 1]
    auto num_valid = z_mask.to(torch::kInt64).sum(1), num_fg_all = seg.to(torch::kInt64).sum(1);    // [B, 1], [B, K]
    auto num_fg_base = at::clamp(num_fg_all, 0, num_min), num_fg_remain = num_fg_all - num_fg_base; // [B, K], [B, K]
    auto num_fg_sampled =
        at::ceil((num_fg_base + (num_fg - num_fg_base.sum(1, true)) * num_fg_remain / (num_fg_remain.sum(1, true) + 1e-5))).to(torch::kInt64); // [B, K]
    auto num_bg_sampled = (num - num_fg_sampled.sum(1, true)).to(torch::kInt64);                                                               // [B, 1]
    auto all_pts = at::cat({num_fg_sampled, num_bg_sampled}, -1);                                                                              // [B, K + 1]
    auto all_sign = at::cat({seg, bg}, -1);                                                                                                    // [B, K + 1]
    // std::cout << num_fg_sampled.sum(1) << " " << num_bg_sampled.sum(1) << std::endl;
    // std::cout << all_sign.any(-1).sum(1) << std::endl;
    // std::cout << all_sign.sum(1) << std::endl;
    // std::cout << all_pts << std::endl;

    // print("%ld %ld %ld %ld\n", all_sign.dim(), all_sign.size(0), all_sign.size(1), all_sign.size(2));
    // printf("%ld %ld %ld %ld\n", all_sign.dim(), all_sign.size(0), all_sign.size(1), all_sign.size(2));
    // printf("%ld %ld %ld %ld %ld %ld %ld %ld\n", num_pts.size(0), num_pts.size(1), num_base.size(0), num_base.size(1), remain_pts.size(0), remain_pts.size(1),
    // tgt_pts.size(0), tgt_pts.size(1));

    /*
        tot_pts, target_mask_pts, min_pts = 1200, 800, 50

        base_num = minimum(num_pts, min_pts)
        remain_pts = num_pts - base_num
        tgt_pts = base_num + (target_mask_pts - base_num.sum()) * remain_pts // remain_pts.sum()
        back_pts = tot_pts - tgt_pts.sum()

        bk_seg = ~seg.any(-1, keepdims=True)
        seg_all = concat([seg, bk_seg], axis=-1)
        num_all = seg_all.sum(-1)
        tgt_pts = concat([tgt_pts, np.array([back_pts])], axis=-1)

        chosen_index = []
        for i in range(seg_all.shape[1]):
            if num_all[i] == 0:
                continue
            cur_seg = np.where(seg_all[:, i])[0]
            np.random.shuffle(cur_seg)
            shuffle_indices = cur_seg[: tgt_pts[i]]
            chosen_index.append(shuffle_indices)
    */
    TORCH_CHECK((rand_idx.max() < N).all().item<bool>() && (rand_idx.min() >= 0).all().item<bool>(), "Bug inside the random permutation!");

    dim3 blocks(B), threads(THREADS_PER_BLOCK);
    maniskill_downsample_kernel<<<blocks, threads, 0, stream>>>(all_pts.data_ptr<int64_t>(), all_sign.data_ptr<bool>(), mask.data_ptr<bool>(),
                                                                index.data_ptr<int64_t>(), rand_idx.data_ptr<int64_t>(), B, N, int(all_pts.size(1)), num,
                                                                min_z);

    AT_CUDA_CHECK(cudaGetLastError());

    return std::make_tuple(index, mask);
}
