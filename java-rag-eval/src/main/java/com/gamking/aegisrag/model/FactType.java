package com.gamking.aegisrag.model;

import java.util.Locale;

/**
 * 事实类型枚举：将校验从单一"查数字"泛化为多类型命题校验。
 */
public enum FactType {
    /** 标量事实：70%、1500元、200元 */
    NUMERIC("numeric"),
    
    /** 实体关系：A药禁止与B药合用 */
    ENTITY_REL("entity_rel"),
    
    /** 限制/时效：仅限2026年前、非直系亲属除外 */
    CONDITION("condition"),
    
    /** 分类/枚举：三级甲等、必须、禁止 */
    CATEGORICAL("categorical");
    
    private final String value;
    
    FactType(String value) {
        this.value = value;
    }
    
    public String getValue() {
        return value;
    }
    
    /**
     * 从字符串解析 FactType，未知类型返回 NUMERIC。
     */
    public static FactType fromString(String type) {
        if (type == null) {
            return NUMERIC;
        }
        return switch (type.toLowerCase(Locale.ROOT)) {
            case "entity_rel" -> ENTITY_REL;
            case "condition" -> CONDITION;
            case "categorical" -> CATEGORICAL;
            default -> NUMERIC;
        };
    }
}