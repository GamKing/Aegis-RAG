package com.gamking.aegisrag;

import com.gamking.aegisrag.text.TextNormalizer;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * TextNormalizer 单元测试
 */
public class TextNormalizerTest {

    @Test
    public void test_normalizeText_basic() {
        assertEquals("1234", TextNormalizer.normalizeText("1,234"));
        assertEquals("hello", TextNormalizer.normalizeText("hello"));
    }

    @Test
    public void test_normalizeText_fullwidth() {
        assertEquals("70%", TextNormalizer.normalizeText("７０％"));
        assertEquals("1500", TextNormalizer.normalizeText("１５００"));
    }

    @Test
    public void test_normalizeForMatch() {
        assertEquals("hello world", TextNormalizer.normalizeForMatch("Hello World"));
        assertEquals("abc123", TextNormalizer.normalizeForMatch("ABC 123"));
    }

    @Test
    public void test_extractTerms_cjk() {
        var terms = TextNormalizer.extractTerms("这是一个测试");
        assertTrue(terms.contains("这是"));
        assertTrue(terms.contains("一个"));
        assertTrue(terms.contains("测试"));
    }

    @Test
    public void test_extractNumbers_basic() {
        var numbers = TextNormalizer.extractNumbers("报销比例为70%，限额1500元");
        assertEquals(2, numbers.size());
        assertEquals("70", numbers.get(0).value());
        assertTrue(numbers.get(0).isPercent());
        assertEquals("1500", numbers.get(1).value());
        assertFalse(numbers.get(1).isPercent());
    }

    @Test
    public void test_extractNumbers_dedup() {
        var numbers = TextNormalizer.extractNumbers("70%的70%与1500元");
        assertEquals(2, numbers.size());
    }

    @Test
    public void test_numberSupportedInContext_match() {
        assertTrue(TextNormalizer.numberSupportedInContext("1500", false, "限额为1500元"));
        assertFalse(TextNormalizer.numberSupportedInContext("1500", false, "限额为15000元"));
        assertFalse(TextNormalizer.numberSupportedInContext("1500", false, "限额为31500元"));
    }

    @Test
    public void test_numberSupportedInContext_percent() {
        assertTrue(TextNormalizer.numberSupportedInContext("70", true, "报销比例为70%"));
        assertFalse(TextNormalizer.numberSupportedInContext("70", true, "费用70元"));
    }

    @Test
    public void test_displayWidth() {
        assertEquals(3, TextNormalizer.displayWidth("abc"));
        assertEquals(4, TextNormalizer.displayWidth("医保"));
        assertEquals(7, TextNormalizer.displayWidth("医保70%"));
    }

    @Test
    public void test_truncateDisplay() {
        String text = "职工医保普通门诊报销比例";
        String truncated = TextNormalizer.truncateDisplay(text, 10);
        assertTrue(TextNormalizer.displayWidth(truncated) <= 10);
        assertTrue(truncated.endsWith("..."));
    }
}
