package org.yah.tools.cuda.api;

import com.sun.jna.Library;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

public interface NVRTC extends Library {

    int NVRTC_SUCCESS = 0;
    int NVRTC_ERROR_COMPILATION = 6;

    int nvrtcVersion(Pointer major, Pointer minor);

    Pointer nvrtcGetErrorString(int result);

    int nvrtcCreateProgram(PointerByReference prog, Pointer src, Pointer name, int numHeaders, Pointer headers, Pointer includeNames);

    int nvrtcDestroyProgram(PointerByReference prog);

    int nvrtcCompileProgram(Pointer prog, int numOptions, Pointer options);

    int nvrtcGetProgramLogSize(Pointer prog, PointerByReference logSizeRet);

    int nvrtcGetProgramLog(Pointer prog, Pointer log);

    int nvrtcGetPTXSize(Pointer prog, PointerByReference ptxSizeRet);

    int nvrtcGetPTX(Pointer prog, Pointer ptx);

    int nvrtcGetCUBINSize(Pointer prog, PointerByReference cubinSizeRet);

    int nvrtcGetCUBIN(Pointer prog, Pointer cubin);

}
