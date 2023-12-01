package org.yah.tools.cuda.proxy.annotations;

import org.yah.tools.cuda.proxy.services.KernelArgumentWriter;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Configure a {@link KernelArgumentWriter} for a parameter
 */
@Target(ElementType.PARAMETER)
@Retention(RetentionPolicy.RUNTIME)
public @interface Writer {
    Class<? extends KernelArgumentWriter<?>> value();
}
