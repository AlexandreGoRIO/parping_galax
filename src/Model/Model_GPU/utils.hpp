#ifndef UTILS_H
#define UTILS_H

#include "cuda.h"
#include <vector>
#include <iostream>

template<typename T>
class CudaBuffer {
public:
    CudaBuffer() {
        m_size = 0;
        m_device_ptr = nullptr;
    }

    CudaBuffer(const std::vector<T>& src) {
        m_size = src.size() * sizeof(T);
        if (cudaMalloc(&m_device_ptr, m_size) != cudaSuccess) {
            m_size = 0;
            std::cout << "error: unable to allocate buffer" << std::endl;
        }
        send(src);
    }

    ~CudaBuffer() {
        cudaFree(&m_device_ptr);
    }

    bool send(const std::vector<T>& src) {
        if (src.size() * sizeof(T) != m_size) {
            std::cout << "error: host and device bufers have different sizes" << std::endl;
            return false;
        }
        if (cudaMemcpy(m_device_ptr, src.data(), m_size, cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cout << "error: unable to copy a buffer to the device" << std::endl;
            return false;
        }
        return true;
    }

    bool retrieve(std::vector<T>& dst) {
        if (dst.size() * sizeof(T) != m_size) {
            std::cout << "error: host and device bufers have different sizes" << std::endl;
            return false;
        }
        if (cudaMemcpy(dst.data(), m_device_ptr, m_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::cout << "error: unable to copy a buffer to the host" << std::endl;
            return false;
        }
        return true;
    }

    T* dev_ptr() {
        return (T*)m_device_ptr;
    }

private:
    size_t m_size;
    void* m_device_ptr;
};

#endif /* UTILS_H */