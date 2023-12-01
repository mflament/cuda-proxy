package org.yah.tools.cuda.support.device;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.support.CUctx_flags;
import org.yah.tools.cuda.support.CudaContextPointer;

import static org.yah.tools.cuda.support.DriverSupport.check;
import static org.yah.tools.cuda.support.DriverSupport.driverAPI;
import static org.yah.tools.cuda.support.NTSHelper.readNTS;

public class DevicePointer extends Pointer {

    private static final int MAX_NAME_SIZE = 512;

    public DevicePointer(PointerByReference ref) {
        super(Pointer.nativeValue(ref.getValue()));
    }

    public String getDeviceName() {
        try (Memory memory = new Memory(MAX_NAME_SIZE)) {
            check(driverAPI().cuDeviceGetName(memory, MAX_NAME_SIZE, this));
            return readNTS(memory, MAX_NAME_SIZE);
        }
    }

    public int[] getDeviceAttributes(CUdevice_attribute... attributes) {
        int[] values = new int[attributes.length];
        try (Memory memory = new Memory(Integer.BYTES)){
            for (int i = 0; i < attributes.length; i++) {
                getDeviceAttribute(memory, attributes[i]);
                values[i] = memory.getInt(0);
            }
            return values;
        }
    }

    public int getDeviceAttribute(CUdevice_attribute attribute) {
        try (Memory memory = new Memory(Integer.BYTES)){
            getDeviceAttribute(memory, attribute);
            return memory.getInt(0);
        }
    }

    public void getDeviceAttribute(Memory dst, CUdevice_attribute attribute) {
        check(driverAPI().cuDeviceGetAttribute(dst, attribute.value(), this));
    }

    public long getTotalMem() {
        PointerByReference bytes = new PointerByReference();
        check(driverAPI().cuDeviceTotalMem(bytes, this));
        return Pointer.nativeValue(bytes.getValue());
    }

    /**
     * @param flags  mask of {@link CUctx_flags}
     * @return new cuContext
     * Note : In most cases it is recommended to use
     * <a href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g9051f2d5c31501997a6cb0530290a300">cuDevicePrimaryCtxRetain</a>.
     */
    public CudaContextPointer createContext(int flags) {
        PointerByReference ptrRef = new PointerByReference();
        check(driverAPI().cuCtxCreate(ptrRef, flags, this));
        return new CudaContextPointer(ptrRef);
    }

    public CudaContextPointer primaryCtxRetain() {
        PointerByReference ptrRef = new PointerByReference();
        check(driverAPI().cuDevicePrimaryCtxRetain(ptrRef, this));
        return new CudaContextPointer(ptrRef);
    }

    public void primaryCtxRelease() {
        check(driverAPI().cuDevicePrimaryCtxRelease(this));
    }
}
