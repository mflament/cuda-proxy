package org.yah.tools.cuda.proxy.annotations;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Create the gridDim from this argument, by using invocation blockDim to compute blocks count from grid threads.
 * Parameter must be an instance of dim3.
 */
@Target(ElementType.PARAMETER)
@Retention(RetentionPolicy.RUNTIME)
public @interface GridThreads {
    /**
     * true if this parameter is also a kernel parameter
     */
    boolean exposed() default false;
}
