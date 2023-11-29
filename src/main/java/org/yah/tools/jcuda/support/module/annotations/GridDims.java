package org.yah.tools.jcuda.support.module.annotations;

import org.yah.tools.jcuda.support.module.Dim;
import org.yah.tools.jcuda.support.module.GridDimsConverter;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.PARAMETER)
@Retention(RetentionPolicy.RUNTIME)
public @interface GridDims {

    Class<? extends GridDimsConverter<?>> converter() default IdentityGridDimsConverter.class;

    boolean exposed() default true;

    final class IdentityGridDimsConverter implements GridDimsConverter<Dim> {
        @Override
        public Dim toGridDim(Dim arg, Dim blockDim) {
            return arg;
        }
    }
}
