import torch
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load
import os

# Get the directory where this file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the CUDA extension
print("Loading CUDA extension... (this may take a minute on first run)")
leaky_relu_cuda = load(
    name="leaky_relu_cuda",
    sources=[os.path.join(current_dir, "leaky_relu_cuda.cu")],
    extra_cuda_cflags=["-O3"],
    verbose=True
)
print("✅ CUDA extension loaded successfully!")


# Custom autograd function
class LeakyReLUFunction(Function):
    """
    Custom CUDA-accelerated Leaky ReLU activation function.
    
    Forward: f(x) = x if x > 0 else alpha * x
    Backward: f'(x) = 1 if x > 0 else alpha
    """
    
    @staticmethod
    def forward(ctx, input, alpha=0.01):
        """
        Forward pass using CUDA kernel.
        
        Args:
            ctx: Context object to save tensors for backward pass
            input: Input tensor
            alpha: Slope for negative values (default: 0.01)
        
        Returns:
            Output tensor after applying Leaky ReLU
        """
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return leaky_relu_cuda.forward(input, alpha)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using CUDA kernel.
        
        Args:
            ctx: Context object with saved tensors
            grad_output: Gradient from next layer
        
        Returns:
            Gradient with respect to input, None for alpha
        """
        input, = ctx.saved_tensors
        grad_input = leaky_relu_cuda.backward(grad_output, input, ctx.alpha)
        return grad_input, None


# PyTorch Module wrapper
class CustomLeakyReLU(nn.Module):
    """
    Custom Leaky ReLU activation layer using CUDA acceleration.
    
    This is a drop-in replacement for nn.LeakyReLU() that uses
    a custom CUDA kernel for GPU acceleration.
    
    Args:
        alpha (float): Slope for negative values (default: 0.01)
    
    Example:
        >>> activation = CustomLeakyReLU(alpha=0.01)
        >>> x = torch.randn(10, 10).cuda()
        >>> y = activation(x)
    """
    
    def __init__(self, alpha=0.01):
        super(CustomLeakyReLU, self).__init__()
        self.alpha = alpha
    
    def forward(self, input):
        """
        Apply custom CUDA Leaky ReLU activation.
        
        Args:
            input: Input tensor (must be on CUDA device)
        
        Returns:
            Activated tensor
        """
        if not input.is_cuda:
            raise RuntimeError("CustomLeakyReLU only supports CUDA tensors. "
                             "Move your tensor to GPU with .cuda() first.")
        
        return LeakyReLUFunction.apply(input, self.alpha)
    
    def extra_repr(self):
        """String representation for print(model)"""
        return f'alpha={self.alpha}'


# Convenience function
def custom_leaky_relu(input, alpha=0.01):
    """
    Functional interface for custom CUDA Leaky ReLU.
    
    Args:
        input: Input tensor (must be on CUDA device)
        alpha: Slope for negative values (default: 0.01)
    
    Returns:
        Activated tensor
    
    Example:
        >>> x = torch.randn(10, 10).cuda()
        >>> y = custom_leaky_relu(x, alpha=0.01)
    """
    return LeakyReLUFunction.apply(input, alpha)


if __name__ == "__main__":
    # Simple test
    print("\n" + "="*50)
    print("Testing Custom CUDA Leaky ReLU")
    print("="*50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. This module requires a GPU.")
        exit(1)
    
    print(f"✅ CUDA is available")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    
    # Create test tensor
    x = torch.randn(5, 5).cuda()
    print(f"\nInput tensor:\n{x}")
    
    # Test the activation
    activation = CustomLeakyReLU(alpha=0.01)
    y = activation(x)
    print(f"\nOutput after Custom Leaky ReLU:\n{y}")
    
    # Compare with PyTorch's built-in
    y_pytorch = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
    print(f"\nPyTorch's Leaky ReLU (for comparison):\n{y_pytorch}")
    
    # Check if they match
    if torch.allclose(y, y_pytorch, rtol=1e-5):
        print("\n✅ Custom CUDA implementation matches PyTorch!")
    else:
        print("\n⚠️  Results differ from PyTorch implementation")
    
    print("\n" + "="*50)
