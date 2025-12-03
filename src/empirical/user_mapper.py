"""
用户类型映射器 (User Type Mapper)

将 Weibo 认证类型映射到模型概念：
- mainstream: 主流媒体（蓝V媒体类）
- wemedia: 自媒体/大V（黄V）
- public: 公众（红V + 无认证）
- other: 其他（蓝V非媒体类）

使用方法：
    mapper = UserTypeMapper()
    user_type = mapper.map_verify_type(verify_typ, user_name)
"""

from __future__ import annotations

import re
from typing import Literal, Optional, Set
from dataclasses import dataclass


# 主流媒体关键词（用于识别蓝V中的媒体账号）
MAINSTREAM_KEYWORDS = {
    # 中央媒体
    "人民日报", "新华社", "央视", "CCTV", "中国日报", "光明日报", "经济日报",
    "中央广播", "环球时报", "参考消息", "中国新闻网", "中国青年报",
    # 地方媒体
    "新京报", "澎湃新闻", "南方都市报", "南方周末", "北京青年报", "北京晚报",
    "上海观察", "解放日报", "文汇报", "新民晚报", "广州日报", "羊城晚报",
    "钱江晚报", "都市快报", "楚天都市报", "华西都市报", "成都商报",
    "大河报", "齐鲁晚报", "半岛都市报", "辽沈晚报", "新文化报",
    # 专业媒体
    "财新", "第一财经", "21世纪经济报道", "经济观察报", "证券时报",
    "健康报", "医学界", "丁香园", "健康时报",
    # 通用后缀
    "日报", "晚报", "时报", "周刊", "新闻", "电视台", "广播",
}

# 政府/官方机构关键词
GOVERNMENT_KEYWORDS = {
    "发布", "政府", "公安", "卫健委", "疾控中心", "CDC", "卫生健康",
    "市场监管", "应急管理", "外交部", "国务院", "人大", "政协",
}


@dataclass
class UserTypeResult:
    """用户类型映射结果"""
    user_type: Literal["mainstream", "wemedia", "public", "government", "other"]
    confidence: float  # 置信度
    reason: str  # 判断理由


class UserTypeMapper:
    """
    用户类型映射器
    
    根据 Weibo 认证类型和用户名判断用户在模型中的角色。
    
    Parameters
    ----------
    custom_mainstream : Set[str], optional
        自定义主流媒体名单（精确匹配）
    custom_wemedia : Set[str], optional
        自定义自媒体/大V名单（精确匹配）
    """
    
    def __init__(
        self,
        custom_mainstream: Optional[Set[str]] = None,
        custom_wemedia: Optional[Set[str]] = None,
    ):
        self.custom_mainstream = custom_mainstream or set()
        self.custom_wemedia = custom_wemedia or set()
        
        # 编译正则
        self.mainstream_pattern = re.compile(
            "|".join(re.escape(kw) for kw in MAINSTREAM_KEYWORDS),
            re.IGNORECASE
        )
        self.government_pattern = re.compile(
            "|".join(re.escape(kw) for kw in GOVERNMENT_KEYWORDS),
            re.IGNORECASE
        )
    
    def map_verify_type(
        self,
        verify_typ: str,
        user_name: str = "",
    ) -> UserTypeResult:
        """
        映射用户类型
        
        Parameters
        ----------
        verify_typ : str
            Weibo 认证类型（蓝V认证/黄V认证/红V认证/无认证）
        user_name : str
            用户名（用于进一步判断蓝V的具体类型）
            
        Returns
        -------
        UserTypeResult
            用户类型及置信度
        """
        verify_typ = str(verify_typ).strip() if verify_typ else ""
        user_name = str(user_name).strip() if user_name else ""
        
        # 1. 自定义名单优先
        if user_name in self.custom_mainstream:
            return UserTypeResult("mainstream", 1.0, "自定义主流媒体名单")
        if user_name in self.custom_wemedia:
            return UserTypeResult("wemedia", 1.0, "自定义自媒体名单")
        
        # 2. 根据认证类型判断
        if "蓝V" in verify_typ or "蓝v" in verify_typ.lower():
            # 蓝V需要进一步判断
            if self.mainstream_pattern.search(user_name):
                return UserTypeResult("mainstream", 0.9, f"蓝V + 媒体关键词匹配")
            elif self.government_pattern.search(user_name):
                return UserTypeResult("government", 0.9, f"蓝V + 政府关键词匹配")
            else:
                return UserTypeResult("other", 0.7, "蓝V但未识别为媒体/政府")
        
        elif "黄V" in verify_typ or "黄v" in verify_typ.lower():
            return UserTypeResult("wemedia", 0.95, "黄V认证（高影响力个人）")
        
        elif "红V" in verify_typ or "红v" in verify_typ.lower():
            return UserTypeResult("public", 0.9, "红V认证（普通个人）")
        
        else:
            # 无认证或其他
            return UserTypeResult("public", 0.85, "无认证用户")
    
    def is_media_source(self, user_type: str) -> bool:
        """判断是否为信息源（媒体）"""
        return user_type in ("mainstream", "wemedia", "government")
    
    def is_public(self, user_type: str) -> bool:
        """判断是否为公众"""
        return user_type in ("public",)


def map_user_types_batch(
    verify_types: list,
    user_names: list,
    mapper: Optional[UserTypeMapper] = None,
) -> list:
    """
    批量映射用户类型
    
    Parameters
    ----------
    verify_types : list
        认证类型列表
    user_names : list
        用户名列表
    mapper : UserTypeMapper, optional
        映射器实例
        
    Returns
    -------
    list
        用户类型列表
    """
    if mapper is None:
        mapper = UserTypeMapper()
    
    results = []
    for vt, un in zip(verify_types, user_names):
        result = mapper.map_verify_type(vt, un)
        results.append(result.user_type)
    
    return results

