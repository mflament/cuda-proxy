package org.yah.tools.jcuda.support.module;

import java.util.Objects;

public class dim3 {

    private final int x;
    private final int y;
    private final int z;

    public dim3(int x) {
        this(x, 1, 1);
    }

    public dim3(int x, int y) {
        this(x, y, 1);
    }

    public dim3(int[] dims) {
        this(dims.length > 0 ? dims[0] : 1, dims.length > 1 ? dims[1] : 1, dims.length > 2 ? dims[2] : 1);
    }

    public dim3(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public int x() {
        return x;
    }

    public int y() {
        return y;
    }

    public int z() {
        return z;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        dim3 dim3 = (dim3) o;
        return x == dim3.x && y == dim3.y && z == dim3.z;
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y, z);
    }

    @Override
    public String toString() {
        return "Dim{" +
                "x=" + x +
                ", y=" + y +
                ", z=" + z +
                '}';
    }

    public static dim3 fromThreads(dim3 gridThreads, dim3 blockDim) {
        return new dim3(blocks(gridThreads.x, blockDim.x), blocks(gridThreads.y, blockDim.y), blocks(gridThreads.z, blockDim.z));
    }

    /**
     * <a href="https://stackoverflow.com/a/2745086/1219319">fast ceiling of threads / blocksDim</a>
     *
     * @param threads  total grid threads
     * @param blockDim block dim
     * @return ceil(threads / blockDim) = gridDim
     */
    public static int blocks(int threads, int blockDim) {
        return 1 + ((threads - 1) / blockDim);
    }

}
