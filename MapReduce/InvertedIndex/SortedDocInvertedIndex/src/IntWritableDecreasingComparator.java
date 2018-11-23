package main.java.SortedDocInvertedIndex;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.WritableComparable;

public class IntWritableDecreasingComparator extends DoubleWritable.Comparator {
    @Override
    public int compare(WritableComparable a, WritableComparable b) {
        return -super.compare(a, b);
    }

    @Override
    public int compare(byte[] b1, int s1, int i1, byte[] b2, int s2, int i2) {
        return -super.compare(b1, s1, i1, b2, s2, i2);
    }
}
