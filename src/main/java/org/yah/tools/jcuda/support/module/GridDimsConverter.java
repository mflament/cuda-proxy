package org.yah.tools.jcuda.support.module;

public interface GridDimsConverter<T> {
    // return an unsigned int
    Dim toGridDim(T arg, Dim blockDim);
}
