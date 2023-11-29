package org.yah.tools.jcuda.support;

import com.sun.jna.Pointer;
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

}
