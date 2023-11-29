package org.yah.tools.jcuda.support.program;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.jcuda.support.NVRTCSupport;

public class CudaProgramPointer extends PointerByReference {

    public CudaProgramPointer() {
    }

    public Memory getPTX() {
        Memory memory = new Memory(getPTXSize());
        NVRTCSupport.check(NVRTCSupport.nvrtc().nvrtcGetPTX(getValue(), memory));
        return memory;
    }

    public long getPTXSize() {
        PointerByReference sizeRet = new PointerByReference();
        NVRTCSupport.check(NVRTCSupport.nvrtc().nvrtcGetPTXSize(getValue(), sizeRet));
        return Pointer.nativeValue(sizeRet.getValue());
    }

    public void destroy() {
        NVRTCSupport.check(NVRTCSupport.nvrtc().nvrtcDestroyProgram(getPointer()));
    }
}
