package org.yah.tools.cuda.proxy.annotations;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface Kernel {

    /**
     * Name of the cuda function to bind. Default to the annotated method name.
     */
    String name() default "";

    /**
     * Static block dimension to use.
     * If a parameter is annotated with @BlockDim, it will be used instead
     */
    int[] blockDim() default {};

    /**
     * Static grid dimension to use.
     * If a parameter is annotated with @GridDim or @GridThreads, it will be used instead
     */
    int[] gridDim() default {};

    /**
     * Static grid dimension in threads.
     * If a parameter is annotated with @GridDim or @GridThreads, it will be used instead
     */
    int[] gridThreads() default {};

    /**
     * Static shared memory size in bytes.
     * If a parameter is annotated with @SharedMemory, it will be used instead
     */
    int sharedMemory() default 0;

}
