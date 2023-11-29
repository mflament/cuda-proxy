package org.yah.tools.jcuda.support.module.annotations;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface Kernel {

    String name() default "";

    int[] blockDim() default {32, 32, 1};

    int sharedMemory() default 0;

}
