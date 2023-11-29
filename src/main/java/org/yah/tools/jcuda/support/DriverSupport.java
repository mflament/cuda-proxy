package org.yah.tools.jcuda.support;

import com.sun.jna.Memory;
import com.sun.jna.ptr.PointerByReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yah.tools.jcuda.jna.DriverAPI;

import javax.annotation.Nullable;

import static org.yah.tools.jcuda.support.NTSHelper.readNTS;

public class DriverSupport {

    private static final Logger LOGGER = LoggerFactory.getLogger(DriverSupport.class);
    private static final int MAX_DEVICE_NAME_LENGTH = 512;

    public static final class cuCtxFlag {
        private cuCtxFlag() {
        }

        public static int CU_CTX_SCHED_AUTO = 0x00;
        public static int CU_CTX_SCHED_SPIN = 0x01;
        public static int CU_CTX_SCHED_YIELD = 0x02;
        public static int CU_CTX_SCHED_BLOCKING_SYNC = 0x04;
        public static int CU_CTX_BLOCKING_SYNC = 0x04;
        public static int CU_CTX_SCHED_MASK = 0x07;
        public static int CU_CTX_MAP_HOST = 0x08;
        public static int CU_CTX_LMEM_RESIZE_TO_MAX = 0x10;
        public static int CU_CTX_COREDUMP_ENABLE = 0x20;
        public static int CU_CTX_USER_COREDUMP_ENABLE = 0x40;
        public static int CU_CTX_SYNC_MEMOPS = 0x80;
        public static int CU_CTX_FLAGS_MASK = 0xFF;
    }

    @Nullable
    private static DriverAPI driverAPI;

    public static synchronized DriverAPI driverAPI() {
        if (driverAPI == null) {
            driverAPI = CudaSupport.createDriverAPI();
            int error = driverAPI().cuInit(0);
            if (error != 0)
                throw new CudaException("nvcuda", error, "?");
        }
        return driverAPI;
    }

    public static CudaContextPointer createContext(int device, int flags) {
        PointerByReference devicePtr = new PointerByReference();
        check(driverAPI().cuDeviceGet(devicePtr, device));
        if (LOGGER.isInfoEnabled()) {
            try (Memory namePtr = new Memory(MAX_DEVICE_NAME_LENGTH)) {
                check(driverAPI().cuDeviceGetName(namePtr, MAX_DEVICE_NAME_LENGTH, devicePtr.getValue()));
                String name = readNTS(namePtr, MAX_DEVICE_NAME_LENGTH);
                LOGGER.info("Using device {} : {}", device, name);
            }
        }

        CudaContextPointer cudaContextPointer = new CudaContextPointer();
        check(driverAPI().cuCtxCreate(cudaContextPointer, flags, devicePtr.getValue()));
        return cudaContextPointer;
    }

    public static void check(int status) {
        if (status != 0) {
            PointerByReference strRef = new PointerByReference();
            driverAPI().cuGetErrorName(status, strRef);
            String error = readNTS(strRef.getValue(), 256);
            throw new CudaException("nvcuda", status, error);
        }
    }

}
