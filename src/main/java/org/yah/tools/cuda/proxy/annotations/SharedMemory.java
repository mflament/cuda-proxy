package org.yah.tools.cuda.proxy.annotations;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Extract kernel shared memory size from a parameter
 * The argument must be an int or a long
 */
@Target(ElementType.PARAMETER)
@Retention(RetentionPolicy.RUNTIME)
public @interface SharedMemory {
    /**
     * true if this parameter is also a kernel parameter
     */
    boolean exposed() default false;
}
