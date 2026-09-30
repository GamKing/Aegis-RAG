package com.gamking.aegisrag.text;

import java.text.Normalizer;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 文本归一化、分词与数值匹配的底层工具集。
 * 
 * 设计约束：
 * - 零第三方依赖（仅标准库），保证评测逻辑可复现、可审计；
 * - 面向中文语料：关键词召回采用 CJK 二元组（bigram）覆盖法；数值忠实性校验
 *   采用边界敏感的正则匹配，避免 "1500" 误命中 "15000" 一类的前缀串扰；
 * - 附带终端显示宽度工具（East Asian Wide/Fullwidth 记 2 列），保证 ASCII
 *   报表在含中文时仍然对齐。
 */
public final class TextNormalizer {

    private TextNormalizer() {}

    // ---------------------------------------------------------------------------
    // 归一化
    // ---------------------------------------------------------------------------

    /** 全角 -> 半角映射表 */
    private static final Map<Character, Character> FULLWIDTH_MAP = new HashMap<>();
    static {
        String fullwidth = "０１２３４５６７８９"
                + "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
                + "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
                + "％．";
        String halfwidth = "0123456789"
                + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                + "abcdefghijklmnopqrstuvwxyz"
                + "%.";
        for (int i = 0; i < fullwidth.length(); i++) {
            FULLWIDTH_MAP.put(fullwidth.charAt(i), halfwidth.charAt(i));
        }
    }

    /** 千分位逗号正则 */
    private static final Pattern THOUSANDS_COMMA_RE = Pattern.compile("(?<=\\d),(?=\\d)");

    /**
     * NFC 归一化 + 全角转半角 + 去掉数字中的千分位逗号（1,500 -> 1500）。
     */
    public static String normalizeText(String text) {
        if (text == null) return "";
        text = Normalizer.normalize(text, Normalizer.Form.NFC);
        StringBuilder sb = new StringBuilder(text.length());
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            sb.append(FULLWIDTH_MAP.getOrDefault(c, c));
        }
        text = sb.toString();
        text = THOUSANDS_COMMA_RE.matcher(text).replaceAll("");
        return text;
    }

    /**
     * 实体/子串匹配用的归一化：全角转半角、去全部空白、忽略大小写。
     */
    public static String normalizeForMatch(String text) {
        return normalizeText(text).replaceAll("\\s+", "").toLowerCase();
    }

    // ---------------------------------------------------------------------------
    // 切分
    // ---------------------------------------------------------------------------

    /** CJK 连续段正则 */
    private static final Pattern CJK_RUN_RE = Pattern.compile("[\\u4e00-\\u9fff]+");

    /** ASCII token 正则（字母词或数字/百分比） */
    private static final Pattern ASCII_TOKEN_RE = Pattern.compile("[A-Za-z]+|\\d+(?:\\.\\d+)?%?");

    /** 默认停用词 */
    public static final Set<String> DEFAULT_STOPWORDS = Set.of(
            "我们", "你们", "他们", "可以", "以及", "如果", "但是", "对于", "根据",
            "通过", "并且", "或者", "还是", "就是", "这个", "那个", "什么", "怎么",
            "哪些", "是否", "如下",
            "the", "a", "an", "of", "to", "is", "are", "for", "and", "or",
            "in", "on", "at", "by", "with"
    );

    /**
     * 抽取召回评测词项：CJK 连续段切 bigram + ASCII 字母词/数字（含 %）。
     */
    public static Set<String> extractTerms(String text) {
        return extractTerms(text, DEFAULT_STOPWORDS);
    }

    public static Set<String> extractTerms(String text, Set<String> stopwords) {
        text = normalizeText(text);
        Set<String> terms = new LinkedHashSet<>();
        Matcher cjkMatcher = CJK_RUN_RE.matcher(text);
        while (cjkMatcher.find()) {
            String run = cjkMatcher.group();
            for (int i = 0; i < run.length() - 1; i++) {
                String bigram = run.substring(i, i + 2);
                if (!stopwords.contains(bigram)) {
                    terms.add(bigram);
                }
            }
        }
        Matcher asciiMatcher = ASCII_TOKEN_RE.matcher(text);
        while (asciiMatcher.find()) {
            String token = asciiMatcher.group().toLowerCase();
            if (!stopwords.contains(token)) {
                terms.add(token);
            }
        }
        return terms;
    }

    // ---------------------------------------------------------------------------
    // 数值忠实性
    // ---------------------------------------------------------------------------

    /** 数值正则 */
    private static final Pattern NUMBER_RE = Pattern.compile("(?<value>\\d+(?:\\.\\d+)?)(?<pct>%)?");

    /**
     * 提取文本中的数值单元，返回去重保序的 [(数值字符串, 是否百分比), ...]。
     */
    public static List<NumberUnit> extractNumbers(String text) {
        text = normalizeText(text);
        List<NumberUnit> seen = new ArrayList<>();
        Matcher matcher = NUMBER_RE.matcher(text);
        while (matcher.find()) {
            String value = matcher.group("value");
            boolean isPct = matcher.group("pct") != null;
            NumberUnit unit = new NumberUnit(value, isPct);
            if (!seen.contains(unit)) {
                seen.add(unit);
            }
        }
        return seen;
    }

/**
     * 边界敏感地判断数值是否在上下文中出现。
     * 
     * 严格规则：
     * 1. 数字边界："1500" 不得命中 "15000" / "31500" 中的子串；
     * 2. 单位一致：答案中的百分数必须以上下文中的百分数形式出现
     *    （防止 "85%" 被 "85元" 洗白）。
     */
    public static boolean numberSupportedInContext(String value, boolean isPercent, String context) {
        context = normalizeText(context);
        String esc = Pattern.quote(value);
        String pattern;
        if (isPercent) {
            pattern = "(?<![\\d.])" + esc + "(?![\\d.])\\s*%";
        } else {
            pattern = "(?<![\\d.])" + esc + "(?![\\d.])";
        }
        return Pattern.compile(pattern).matcher(context).find();
    }

    /** 数值单元：值 + 是否百分比 */
    public record NumberUnit(String value, boolean isPercent) {
        public NumberUnit {
            if (value == null || value.isBlank()) {
                throw new IllegalArgumentException("value must not be blank");
            }
        }
    }

    // ---------------------------------------------------------------------------
    // 终端显示宽度（CJK 对齐）
    // ---------------------------------------------------------------------------

    /**
     * 终端显示宽度：East Asian Wide/Fullwidth 记 2，其余记 1。
     */
    public static int displayWidth(String text) {
        if (text == null) return 0;
        int width = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            String eaw = Character.toString(c);
            // 简化判断：CJK 字符记 2，其余记 1
            if (c >= 0x4e00 && c <= 0x9fff) {
                width += 2;
            } else {
                width += 1;
            }
        }
        return width;
    }

    /**
     * 按显示宽度截断并追加省略后缀。
     */
    public static String truncateDisplay(String text, int width, String suffix) {
        if (text == null) text = "";
        if (suffix == null) suffix = "...";
        if (displayWidth(text) <= width) {
            return text;
        }
        int budget = Math.max(width - displayWidth(suffix), 0);
        StringBuilder out = new StringBuilder();
        int acc = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            int w = (c >= 0x4e00 && c <= 0x9fff) ? 2 : 1;
            if (acc + w > budget) {
                break;
            }
            out.append(c);
            acc += w;
        }
        return out.toString() + suffix;
    }

    public static String truncateDisplay(String text, int width) {
        return truncateDisplay(text, width, "...");
    }

    // ---------------------------------------------------------------------------
    // 通用事实校验辅助
    // ---------------------------------------------------------------------------

    /** 否定词集合：用于检测事实是否包含否定/例外条件 */
    public static final Set<String> NEGATION_WORDS = Set.of(
            "不得", "禁止", "免予", "除外", "不纳入", "不适用", "不可", "不能",
            "不允许", "不应", "无须", "无需", "非", "无", "不", "否"
    );

    /** 时效/条件关键词：用于识别限制类事实 */
    public static final Set<String> CONDITION_KEYWORDS = Set.of(
            "仅限", "仅", "必须", "须", "应当", "应", "需", "需要",
            "之前", "之后", "以内", "以外", "范围内", "范围外",
            "前提", "条件", "例外", "特殊", "除外"
    );

    /**
     * 检测文本是否包含否定词。
     */
    public static boolean detectNegation(String text) {
        if (text == null) return false;
        String normalized = normalizeText(text);
        for (String word : NEGATION_WORDS) {
            if (normalized.contains(word)) {
                return true;
            }
        }
        return false;
    }

    /**
     * 提取文本中的条件/时效关键词。
     */
    public static List<String> extractConditionKeywords(String text) {
        if (text == null) return List.of();
        String normalized = normalizeText(text);
        List<String> found = new ArrayList<>();
        for (String kw : CONDITION_KEYWORDS) {
            if (normalized.contains(kw)) {
                found.add(kw);
            }
        }
        return found;
    }

    /**
     * 判断两个实体是否在同一窗口内共现。
     * 
     * @param entityA 第一个实体
     * @param entityB 第二个实体
     * @param context 上下文文本
     * @param sentenceWindow true 表示句子级，false 表示段落级
     */
    public static boolean entitiesCoOccur(String entityA, String entityB, String context, boolean sentenceWindow) {
        if (entityA == null || entityB == null || entityA.isBlank() || entityB.isBlank()) {
            return false;
        }
        String normA = normalizeForMatch(entityA);
        String normB = normalizeForMatch(entityB);

        if (sentenceWindow) {
            // 句子级共现
            String[] sentences = context.split("[。；\\n]");
            for (String sent : sentences) {
                String norm = normalizeForMatch(sent);
                if (norm.contains(normA) && norm.contains(normB)) {
                    return true;
                }
            }
            return false;
        } else {
            // 段落级共现
            String[] paragraphs = context.split("\\n\\n");
            for (String para : paragraphs) {
                String norm = normalizeForMatch(para);
                if (norm.contains(normA) && norm.contains(normB)) {
                    return true;
                }
            }
            return false;
        }
    }
}
