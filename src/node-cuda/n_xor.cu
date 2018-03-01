/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "cuda_utils.h"
#include "../node/n_xor.h"

namespace FT{
   		
    __global__ void Xor(bool * x1, bool * x2, bool * out, size_t N)
    {                    
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            out[i] = x1[i] != x2[i];
        }
        return;
    }
    /// Evaluates the node and updates the stack states. 
    void NodeXor::evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
            vector<ArrayXb>& stack_b)
    {
        ArrayXb x2 = stack_b.back(); stack_b.pop_back();
        ArrayXb x1 = stack_b.back(); stack_b.pop_back();
        // evaluate on the GPU
        ArrayXb result = ArrayXb(x1.size());
        size_t N = result.size();
        bool * dev_res;
        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        // allocate device arrays
        bool * dev_x1, * dev_x2 ; 
        HANDLE_ERROR(cudaMalloc((void **)& dev_x1, sizeof(bool)*N));
        HANDLE_ERROR(cudaMalloc((void **)& dev_x2, sizeof(bool)*N));
        HANDLE_ERROR(cudaMalloc((void **)&dev_res, sizeof(bool)*N));
        // Copy to device
        HANDLE_ERROR(cudaMemcpy(dev_x1, x1.data(), sizeof(bool)*N, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_x2, x2.data(), sizeof(bool)*N, cudaMemcpyHostToDevice));

        Xor<<< 32*numSMs, 128 >>>(dev_x1, dev_x2, dev_res, N);
       
        // Copy to host
        HANDLE_ERROR(cudaMemcpy(result.data(), dev_res, sizeof(bool)*N, cudaMemcpyDeviceToHost));
        
        stack_b.push_back(limited(result));
        // Free memory
        cudaFree(dev_x1); cudaFree(dev_x2); cudaFree(dev_res);
    }

}	


