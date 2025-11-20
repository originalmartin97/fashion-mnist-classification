#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for custom Leaky ReLU activation
// Formula: f(x) = x if x > 0, else alpha * x
__global__ void leaky_relu_forward_kernel(
    const float* input,
    float* output,
    const int size,
    const float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x > 0 ? x : alpha * x;
    }
}

// CUDA kernel for backward pass (gradient computation)
__global__ void leaky_relu_backward_kernel(
    const float* grad_output,
    const float* input,
    float* grad_input,
    const int size,
    const float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        grad_input[idx] = x > 0 ? grad_output[idx] : alpha * grad_output[idx];
    }
}

// C++ wrapper for forward pass
torch::Tensor leaky_relu_cuda_forward(
    torch::Tensor input,
    float alpha
) {
    auto output = torch::zeros_like(input);
    
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    leaky_relu_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        alpha
    );
    
    return output;
}

// C++ wrapper for backward pass
torch::Tensor leaky_relu_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    float alpha
) {
    auto grad_input = torch::zeros_like(input);
    
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    leaky_relu_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        size,
        alpha
    );
    
    return grad_input;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_cuda_forward, "Leaky ReLU forward (CUDA)");
    m.def("backward", &leaky_relu_cuda_backward, "Leaky ReLU backward (CUDA)");
}
