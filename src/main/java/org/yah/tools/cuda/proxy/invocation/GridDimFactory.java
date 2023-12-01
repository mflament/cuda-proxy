package org.yah.tools.cuda.proxy.invocation;

import org.yah.tools.cuda.proxy.dim3;

public interface GridDimFactory {
    dim3 createGridDim(Object[] args, dim3 blockDim);
}
