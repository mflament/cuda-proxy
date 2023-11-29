package org.yah.tools.jcuda.jna;

import com.sun.jna.Library;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

public interface RuntimeAPI extends Library {

    int CUDA_SUCCESS = 0;

    enum cudaMemcpyKind {
        /**
         * < Host   -> Host
         */
        cudaMemcpyHostToHost,
        /**
         * < Host   -> Device
         */
        cudaMemcpyHostToDevice,
        /**
         * < Device -> Host
         */
        cudaMemcpyDeviceToHost,
        /**
         * < Device -> Device
         */
        cudaMemcpyDeviceTo,
        /**
         * < Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing
         */
        cudaMemcpyDefaultDevice,
    }

    int cudaInitDevice(int device, int deviceFlags, int flags);

    int cudaRuntimeGetVersion(Pointer runtimeVersion);

    int cudaDriverGetVersion(Pointer driverVersion);

    int cudaGetDevice(Pointer device);

    int cudaSetDevice(int device);

    int cudaGetDeviceCount(Pointer count);

    int cudaGetDeviceProperties(Pointer prop, int device);

    int cudaDeviceReset();

    // device management https://docs.nvidia.com/cuda/archive/11.7.0/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE
    int cudaDeviceSynchronize();

    Pointer cudaGetErrorString(int error);

    // memory management https://docs.nvidia.com/cuda/archive/11.7.0/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY
    int cudaMalloc(PointerByReference devPtr, long size);

    int cudaFree(Pointer devPtr);

    int cudaMemcpy(Pointer dst, Pointer src, long count, int kind);

    int cudaMemset(Pointer devPtr, int value, long count);

    int cudaMemGetInfo(PointerByReference free, PointerByReference total);

    int cudaHostRegister(PointerByReference ptr, long size, int flags);

    int cudaHostUnregister(Pointer ptr);

    int cudaMallocHost(PointerByReference ptr, long size);

    int cudaHostGetDevicePointer(PointerByReference pDevice, Pointer pHost, int flags);

}
