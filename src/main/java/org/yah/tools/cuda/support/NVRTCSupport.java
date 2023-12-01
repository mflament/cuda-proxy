package org.yah.tools.cuda.support;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.api.NVRTC;
import org.yah.tools.cuda.support.program.CudaProgramPointer;

import javax.annotation.Nullable;

import static org.yah.tools.cuda.support.NTSHelper.readNTS;

public class NVRTCSupport {
    @Nullable
    private static NVRTC nvrtc;

    public static synchronized NVRTC nvrtc() {
        if (nvrtc == null)
            nvrtc = CudaSupport.createNVRTC();
        return nvrtc;
    }

    public static String getProgramLog(CudaProgramPointer prog) {
        PointerByReference logSizeRet = new PointerByReference();
        NVRTCSupport.check(NVRTCSupport.nvrtc().nvrtcGetProgramLogSize(prog, logSizeRet));
        long size = Pointer.nativeValue(logSizeRet.getValue());
        try (Memory log = new Memory(size)) {
            NVRTCSupport.check(NVRTCSupport.nvrtc().nvrtcGetProgramLog(prog, log));
            return readNTS(log, size);
        }
    }

    public static void check(int status) {
        if (status != 0) {
            Pointer pointer = nvrtc().nvrtcGetErrorString(status);
            String error = readNTS(pointer, 256);
            throw new CudaException("nvrtc", status, error);
        }
    }

}
