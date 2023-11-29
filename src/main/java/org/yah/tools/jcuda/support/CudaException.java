package org.yah.tools.jcuda.support;

public class CudaException extends RuntimeException {
    private final String module;
    private final int status;

    public CudaException(String module, int status, String errorName) {
        super(String.format("%s error %d: %s", module, status, errorName));
        this.module = module;
        this.status = status;
    }

    public String getModule() {
        return module;
    }

    public int getStatus() {
        return status;
    }
}
