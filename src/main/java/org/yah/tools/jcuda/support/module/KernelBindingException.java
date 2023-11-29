package org.yah.tools.jcuda.support.module;

public class KernelBindingException extends RuntimeException {
    public KernelBindingException(String message) {
        super(message);
    }

    public KernelBindingException(String message, Throwable cause) {
        super(message, cause);
    }
}
