package org.yah.tools.cuda.support;

public final class CUctx_flags {
    private CUctx_flags() {
    }

    public static int CU_CTX_SCHED_AUTO = 0x00;
    public static int CU_CTX_SCHED_SPIN = 0x01;
    public static int CU_CTX_SCHED_YIELD = 0x02;
    public static int CU_CTX_SCHED_BLOCKING_SYNC = 0x04;
    public static int CU_CTX_BLOCKING_SYNC = 0x04;
    public static int CU_CTX_SCHED_MASK = 0x07;
    public static int CU_CTX_MAP_HOST = 0x08;
    public static int CU_CTX_LMEM_RESIZE_TO_MAX = 0x10;
    public static int CU_CTX_COREDUMP_ENABLE = 0x20;
    public static int CU_CTX_USER_COREDUMP_ENABLE = 0x40;
    public static int CU_CTX_SYNC_MEMOPS = 0x80;
    public static int CU_CTX_FLAGS_MASK = 0xFF;
}
