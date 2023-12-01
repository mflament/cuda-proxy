package org.yah.tools.jcuda.support.module.invocation;

import org.yah.tools.jcuda.support.module.dim3;

public interface BlockDimFactory {
    dim3 createBlockDim(Object[] args);
}
