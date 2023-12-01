package org.yah.tools.jcuda.support.module.invocation;

import org.yah.tools.jcuda.support.module.dim3;

public interface GridDimFactory {
    dim3 createGridDim(Object[] args, dim3 blockDim);
}
