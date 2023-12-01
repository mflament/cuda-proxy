package org.yah.tools.cuda.support.program;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yah.tools.cuda.api.NVRTC;
import org.yah.tools.cuda.support.NVRTCSupport;
import org.yah.tools.cuda.support.device.CUdevice_attribute;
import org.yah.tools.cuda.support.device.DevicePointer;

import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import static org.yah.tools.cuda.support.NTSHelper.writeNTS;
import static org.yah.tools.cuda.support.NVRTCSupport.check;
import static org.yah.tools.cuda.support.NVRTCSupport.nvrtc;

public class CudaProgramBuilder {

    private static final Logger LOGGER = LoggerFactory.getLogger(CudaProgramBuilder.class);

    public static CudaProgramBuilder create(String source) {
        Objects.requireNonNull(source, "source is null");
        return new CudaProgramBuilder(source);
    }

    private final String source;
    @Nullable
    private String programName;
    private final List<String> headers = new ArrayList<>();
    private final List<String> includeNames = new ArrayList<>();
    private final List<String> compileOptions = new ArrayList<>();

    private CudaProgramBuilder(String source) {
        this.source = source;
    }

    public String source() {
        return source;
    }

    @Nullable
    public String programName() {
        return programName;
    }

    public CudaProgramBuilder withProgramName(@Nullable String programName) {
        this.programName = programName;
        return this;
    }

    public List<String> headers() {
        return headers;
    }

    public CudaProgramBuilder withHeaders(String... headers) {
        this.headers.addAll(Arrays.asList(headers));
        return this;
    }

    public List<String> includeNames() {
        return includeNames;
    }

    public CudaProgramBuilder withIncludes(String... includeNames) {
        this.includeNames.addAll(Arrays.asList(includeNames));
        return this;
    }

    public List<String> compileOptions() {
        return compileOptions;
    }

    /**
     * @param options compile options from <a href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__options">...</a>
     */
    public CudaProgramBuilder withCompileOptions(String... options) {
        this.compileOptions.addAll(Arrays.asList(options));
        return this;
    }

    /**
     * Helper to add compile options --gpu-architecture=<arch> (-arch)
     *
     * @param version version : (ie: 50,52, ... 90, 90a)
     * @see <a href="https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/">Matching CUDA arch and CUDA gencode for various NVIDIA architectures</a>
     */
    public CudaProgramBuilder withComputeVersion(String version) {
        return withCompileOptions(String.format("--gpu-architecture=compute_%s", version));
    }

    public CudaProgramBuilder withComputeVersion(DevicePointer device) {
        return withComputeVersion(cc_version(device));
    }

    public CudaProgramBuilder withSMVersion(String version) {
        return withCompileOptions(String.format("--gpu-architecture=sm_%s", version));
    }

    public CudaProgramBuilder withSMVersion(DevicePointer device) {
        return withSMVersion(cc_version(device));
    }

    private String cc_version(DevicePointer device) {
        int[] cc = device.getDeviceAttributes(CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
        return String.format("%d%d", cc[0], cc[1]);
    }

    /**
     * Helper to set compute/sm version from the selected
     */
    public CudaProgramBuilder withDeviceComputeVersion(DevicePointer device) {
        int[] cc = device.getDeviceAttributes(CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
        return withComputeVersion(cc[0] + Integer.toString(cc[1]));
    }

    public CudaProgramBuilder withDeviceSMVersion(DevicePointer device) {
        int[] cc = device.getDeviceAttributes(CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
        return withSMVersion(cc[0] + Integer.toString(cc[1]));
    }

    public CudaProgramPointer build() {
        if (source == null)
            throw new IllegalStateException("No source configured");

        if (headers.size() != includeNames.size())
            throw new IllegalArgumentException("headers size does not match includeNames size: (" + headers + " <> " + includeNames + ")");

        LOGGER.debug("building program {}", this);
        int stackSize = source.length() + 1;
        if (programName != null)
            stackSize += programName.length() + 1;
        stackSize += headers.stream().mapToInt(h -> h.length() + 1).sum();
        stackSize += includeNames.stream().mapToInt(n -> n.length() + 1).sum();

        // 1 NTS per options + 1 pointer to this NTS
        int optionsSize = compileOptions.stream().mapToInt(s -> s.length() + 1).sum() + compileOptions.size() * Native.POINTER_SIZE;
        if (optionsSize > stackSize)
            stackSize = optionsSize;

        try (Memory stack = new Memory(stackSize)) {
            Pointer ptr = writeNTS(stack, source);
            Pointer namePtr = Pointer.NULL, headersPtr = Pointer.NULL, includeNamesPtr = Pointer.NULL;
            if (programName != null) {
                namePtr = ptr;
                ptr = writeNTS(ptr, programName);
            }
            if (!headers.isEmpty()) {
                headersPtr = ptr;
                for (String header : headers) ptr = writeNTS(ptr, header);
                includeNamesPtr = ptr;
                for (String includeName : includeNames) ptr = writeNTS(ptr, includeName);
            }
            PointerByReference ptrRef = new PointerByReference();
            check(nvrtc().nvrtcCreateProgram(ptrRef, stack, namePtr, headers.size(), headersPtr, includeNamesPtr));
            CudaProgramPointer prog = new CudaProgramPointer(ptrRef);

            // one pointer per options string, first in stack
            ptr = stack.share(compileOptions.size() * (long) Native.POINTER_SIZE);
            int optionsIndex = 0;
            for (String compileOption : compileOptions) {
                stack.setPointer(optionsIndex * (long) Native.POINTER_SIZE, ptr);
                ptr = writeNTS(ptr, compileOption);
                optionsIndex++;
            }

            int error = nvrtc().nvrtcCompileProgram(prog, compileOptions.size(), stack);
            if (error == NVRTC.NVRTC_ERROR_COMPILATION) {
                String programLog = NVRTCSupport.getProgramLog(prog);
                prog.destroy();
                throw new BuildProgramException(programLog);
            }
            return prog;
        }
    }

    @Override
    public String toString() {
        return "CudaProgramBuilder{" +
                "source='" + StringUtils.abbreviate(source, 25) + '\'' +
                ", programName='" + programName + '\'' +
                ", headers=" + headers +
                ", includeNames=" + includeNames +
                ", compileOptions=" + compileOptions +
                '}';
    }
}
