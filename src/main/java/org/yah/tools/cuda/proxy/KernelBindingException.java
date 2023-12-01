package org.yah.tools.cuda.proxy;

public class KernelBindingException extends RuntimeException {
    public KernelBindingException(String message) {
        super(message);
    }

    public KernelBindingException(String message, Throwable cause) {
        super(message, cause);
    }
}
