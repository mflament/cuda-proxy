package org.yah.tools.jcuda.support.module.annotations;

import org.yah.tools.jcuda.support.module.KernelArgumentWriter;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.PARAMETER)
@Retention(RetentionPolicy.RUNTIME)
public @interface Writer {
    Class<? extends KernelArgumentWriter<?>> value();
}