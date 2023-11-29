package org.yah.tools.jcuda.support.program;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import org.yah.tools.jcuda.jna.NVRTC;
import org.yah.tools.jcuda.support.NVRTCSupport;

import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import static org.yah.tools.jcuda.support.NTSHelper.writeNTS;

public class CudaProgramBuilder {

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

    public CudaProgramBuilder withCompileOptions(String... options) {
        this.compileOptions.addAll(Arrays.asList(options));
        return this;
    }

    public CudaProgramPointer build() {
        if (source == null)
            throw new IllegalStateException("No source configured");

        if (headers.size() != includeNames.size())
            throw new IllegalArgumentException("headers size does not match includeNames size: (" + headers + " <> " + includeNames + ")");

        int stackSize = source.length() + 1;
        if (programName != null)
            stackSize += programName.length() + 1;
        stackSize += headers.stream().mapToInt(h -> h.length() + 1).sum();
        stackSize += includeNames.stream().mapToInt(n -> n.length() + 1).sum();

        int optionsSize = compileOptions.stream().mapToInt(s -> s.length() + 1).sum();
        if (optionsSize > stackSize)
            stackSize = optionsSize;

        CudaProgramPointer prog = new CudaProgramPointer();
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
            NVRTCSupport.check(NVRTCSupport.nvrtc().nvrtcCreateProgram(prog, stack, namePtr, headers.size(), headersPtr, includeNamesPtr));

            ptr = stack;
            for (String compileOption : compileOptions) {
                ptr = writeNTS(ptr, compileOption);
            }
            int error = NVRTCSupport.nvrtc().nvrtcCompileProgram(prog.getValue(), compileOptions.size(), stack);
            if (error == NVRTC.NVRTC_ERROR_COMPILATION) {
                String programLog = NVRTCSupport.getProgramLog(prog);
                prog.destroy();
                throw new BuildProgramException(programLog);
            }
            return prog;
        }
    }

}
