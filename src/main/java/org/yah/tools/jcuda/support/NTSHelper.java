package org.yah.tools.jcuda.support;

import com.sun.jna.Pointer;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/**
 * Null terminated string helper
 */
public final class NTSHelper {

    public static String readNTS(byte[] bytes) {
        int i;
        for (i = 0; i < bytes.length; i++) {
            if (bytes[i] == 0) break;
        }
        return new String(bytes, 0, i, StandardCharsets.US_ASCII);
    }

    public static String readNTS(Pointer pointer, long maxLength) {
        ByteBuffer byteBuffer = pointer.getByteBuffer(0, maxLength);
        StringBuilder sb = new StringBuilder();
        while (true) {
            byte b = byteBuffer.get();
            if (b == 0)
                break;
            sb.append((char) b);
        }
        return sb.toString();
    }

    public static Pointer writeNTS(Pointer dst, String src) {
        for (int i = 0; i < src.length(); i++) {
            char c = src.charAt(i);
            dst.setByte(i, (byte) c);
        }
        dst.setByte(src.length(), (byte) 0);
        return dst.share(src.length() + 1);
    }

    private NTSHelper() {
    }
}
