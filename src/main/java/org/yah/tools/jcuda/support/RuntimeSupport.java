package org.yah.tools.jcuda.support;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.jcuda.jna.RuntimeAPI;

import javax.annotation.Nullable;

import static org.yah.tools.jcuda.support.NTSHelper.readNTS;

public class RuntimeSupport {

    @Nullable
    private static RuntimeAPI runtimeAPI;

    public static synchronized RuntimeAPI runtimeAPI() {
        if (runtimeAPI == null)
            runtimeAPI = CudaSupport.createRuntimeApi();
        return runtimeAPI;
    }

    public static void check(int status) {
        if (status != 0) {
            Pointer pointer = runtimeAPI().cudaGetErrorString(status);
            String error = readNTS(pointer, 256);
            throw new CudaException("cudart", status, error);
        }
    }

    public static String getRuntimeGetVersion() {
        PointerByReference ptr = new PointerByReference();
        try (Memory memory = new Memory(Integer.BYTES)) {
            check(runtimeAPI().cudaRuntimeGetVersion(memory));
            int ver = memory.getInt(0);
            return String.format("%d.%d", ver / 1000, ver % 1000);
        }
    }
}
