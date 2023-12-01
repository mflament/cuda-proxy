package org.yah.tools.cuda.proxy.services;

import com.sun.jna.Native;
import com.sun.jna.Pointer;
import org.yah.tools.cuda.proxy.dim3;

public final class DefaultWriters {

    public static final class ByteWriter implements KernelArgumentWriter<Byte> {

        public static final ByteWriter INSTANCE = new ByteWriter();

        @Override
        public long size() {
            return Byte.BYTES;
        }

        @Override
        public void write(Byte arg, Pointer dst) {
            dst.setByte(0, arg);
        }
    }

    public static final class ShortWriter implements KernelArgumentWriter<Short> {

        public static final ShortWriter INSTANCE = new ShortWriter();

        @Override
        public long size() {
            return Short.BYTES;
        }

        @Override
        public void write(Short arg, Pointer dst) {
            dst.setShort(0, arg);
        }
    }

    public static final class IntegerWriter implements KernelArgumentWriter<Integer> {

        public static final IntegerWriter INSTANCE = new IntegerWriter();

        @Override
        public long size() {
            return Integer.BYTES;
        }

        @Override
        public void write(Integer arg, Pointer dst) {
            dst.setInt(0, arg);
        }
    }

    public static final class LongWriter implements KernelArgumentWriter<Long> {

        public static final LongWriter INSTANCE = new LongWriter();

        @Override
        public long size() {
            return Long.BYTES;
        }

        @Override
        public void write(Long arg, Pointer dst) {
            dst.setLong(0, arg);
        }
    }

    public static final class FloatWriter implements KernelArgumentWriter<Float> {

        public static final FloatWriter INSTANCE = new FloatWriter();

        @Override
        public long size() {
            return Float.BYTES;
        }

        @Override
        public void write(Float arg, Pointer dst) {
            dst.setFloat(0, arg);
        }
    }

    public static final class DoubleWriter implements KernelArgumentWriter<Double> {

        public static final DoubleWriter INSTANCE = new DoubleWriter();

        @Override
        public long size() {
            return Double.BYTES;
        }

        @Override
        public void write(Double arg, Pointer dst) {
            dst.setDouble(0, arg);
        }
    }

    @SuppressWarnings("rawtypes")
    public static final class EnumWriter implements KernelArgumentWriter<Enum> {

        public static final EnumWriter INSTANCE = new EnumWriter();

        @Override
        public long size() {
            return Integer.BYTES;
        }

        @Override
        public void write(Enum arg, Pointer dst) {
            dst.setInt(0, arg.ordinal());
        }
    }

    public static final class PointerWriter implements KernelArgumentWriter<Pointer> {

        public static final PointerWriter INSTANCE = new PointerWriter();

        @Override
        public long size() {
            return Native.POINTER_SIZE;
        }

        @Override
        public void write(Pointer arg, Pointer dst) {
            dst.setPointer(0, arg);
        }
    }

    public static final class WritableWriter implements KernelArgumentWriter<Writable> {

        public static final WritableWriter INSTANCE = new WritableWriter();

        @Override
        public long size() {
            return Native.POINTER_SIZE;
        }

        @Override
        public void write(Writable arg, Pointer dst) {
            dst.setPointer(0, arg.pointer());
        }
    }

    public static final class Dim3Writer implements KernelArgumentWriter<dim3> {

        public static final Dim3Writer INSTANCE = new Dim3Writer();

        @Override
        public long size() {
            return Integer.BYTES * 3;
        }

        @Override
        public void write(dim3 arg, Pointer dst) {
            dst.setInt(0, arg.x());
            dst.setInt(Integer.BYTES, arg.y());
            dst.setInt(Integer.BYTES * 2, arg.z());
        }
    }

    public static void register(TypeWriterRegistry registry) {
        registry.register(Byte.TYPE, ByteWriter.INSTANCE);
        registry.register(Short.TYPE, ShortWriter.INSTANCE);
        registry.register(Integer.TYPE, IntegerWriter.INSTANCE);
        registry.register(Long.TYPE, LongWriter.INSTANCE);
        registry.register(Float.TYPE, FloatWriter.INSTANCE);
        registry.register(Double.TYPE, DoubleWriter.INSTANCE);
        registry.register(Enum.class, EnumWriter.INSTANCE);
        registry.register(Pointer.class, PointerWriter.INSTANCE);
        registry.register(Writable.class, WritableWriter.INSTANCE);
        registry.register(dim3.class, Dim3Writer.INSTANCE);
    }
}
