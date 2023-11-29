package org.yah.tools.jcuda.support.module.annotations;

import org.yah.tools.jcuda.support.module.Dim;
import org.yah.tools.jcuda.support.module.GridDimConverter;
import org.yah.tools.jcuda.support.module.DimName;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.PARAMETER)
@Retention(RetentionPolicy.RUNTIME)
public @interface GridDim {

    DimName dim();

    Class<? extends GridDimConverter<?>> converter() default IdentityGridDimConverter.class;

    boolean exposed() default true;

    final class IdentityGridDimConverter implements GridDimConverter<Integer> {
        @Override
        public int toDim(Integer arg, Dim blockDim) {
            return arg;
        }
    }
}
