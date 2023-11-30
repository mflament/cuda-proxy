package org.yah.tools.jcuda.jna;

import com.sun.jna.Library;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

public interface DriverAPI extends Library {

    // 6.3. Initialization
    int cuInit(int flags);

    int cuDeviceGetCount(Pointer count);

    int cuDeviceGet(PointerByReference device, int ordinal);

    int cuDeviceGetName(Pointer name, int len, Pointer dev);

    int cuDeviceGetAttribute(Pointer pi, int attrib, Pointer dev);

    int cuDeviceTotalMem(PointerByReference bytes, Pointer dev);

    int cuDriverGetVersion(Pointer driverVersion);

    // 6.8. Context Management
    int cuCtxCreate(PointerByReference pctx, int flags, Pointer dev);

    int cuDevicePrimaryCtxRetain(PointerByReference pctx, Pointer dev);

    int cuDevicePrimaryCtxRelease(Pointer dev);

    int cuCtxDestroy(Pointer ctx);

    int cuCtxGetCurrent(PointerByReference pctx);

    int cuCtxGetDevice(PointerByReference device);

    int cuCtxSetCurrent(Pointer ctx);

    int cuCtxPushCurrent(Pointer ctx);

    int cuCtxSynchronize();

    // 6.10. Module Management
    int cuModuleLoad(PointerByReference module, Pointer fname);

    int cuModuleLoadData(PointerByReference module, Pointer image);

    int cuModuleLoadFatBinary(PointerByReference module, Pointer fatCubin);

    int cuModuleGetFunction(PointerByReference hfunc, Pointer hmod, Pointer name);

    int cuModuleUnload(Pointer hmod);

    // 6.13. Memory Management
    int cuMemAlloc(PointerByReference dptr, long bytesize);

    int cuMemFree(Pointer dptr);

    int cuMemcpyHtoD(Pointer dstDevice, Pointer srcHost, long ByteCount);

    int cuMemcpyDtoH(Pointer dstHost, Pointer srcDevice, long ByteCount);

    int cuMemGetInfo(PointerByReference free, PointerByReference total);

    int cuMemHostRegister(Pointer p, long bytesize, int Flags);

    int cuMemHostUnregister(Pointer p);

    int cuMemHostAlloc(PointerByReference pp, long bytesize, int Flags);

    int cuMemHostGetDevicePointer(PointerByReference pdptr, Pointer p, int Flags);

    int cuMemsetD16(Pointer dstDevice, short us, long N);

    int cuMemsetD32(Pointer dstDevice, int ui, long N);

    int cuMemsetD8(Pointer dstDevice, char uc, long N);

    // 6.22. Execution Control
    int cuLaunchKernel(Pointer f, int gridDimX, int gridDimY, int gridDimZ, int blockDimX, int blockDimY, int blockDimZ, int sharedMemBytes, Pointer hStream, Pointer kernelParams, Pointer extra);

    int cuGetErrorName(int error, PointerByReference pStr);

    int cuGetErrorString(int error, PointerByReference pStr);

}
