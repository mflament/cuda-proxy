package org.yah.tools.jcuda.support.program;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.jcuda.support.NVRTCSupport;

public class CudaProgramPointer extends Pointer {

    public CudaProgramPointer(PointerByReference ref) {
        super(Pointer.nativeValue(ref.getValue()));
    }

    public Memory getPTX() {
        Memory memory = new Memory(getPTXSize());
        NVRTCSupport.check(NVRTCSupport.nvrtc().nvrtcGetPTX(this, memory));
        return memory;
    }

    public long getPTXSize() {
        PointerByReference sizeRet = new PointerByReference();
        NVRTCSupport.check(NVRTCSupport.nvrtc().nvrtcGetPTXSize(this, sizeRet));
        return Pointer.nativeValue(sizeRet.getValue());
    }

    public void destroy() {
        NVRTCSupport.check(NVRTCSupport.nvrtc().nvrtcDestroyProgram(new PointerByReference(this)));
    }
}
