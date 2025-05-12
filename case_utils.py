"""
大小写处理工具模块
提供处理变量名和列名大小写一致性的工具函数
"""

import pandas as pd

def create_case_mapping(df):
    """
    创建列名大小写映射字典
    
    参数:
    df: DataFrame - 包含列名的数据框
    
    返回:
    dict - 小写列名到原始列名的映射
    """
    case_mapping = {}
    for col in df.columns:
        case_mapping[col.lower()] = col
    return case_mapping

def get_column_case_insensitive(df, col_name, default=None):
    """
    以大小写不敏感方式获取数据框中的列名
    
    参数:
    df: DataFrame - 要查找列的数据框
    col_name: str - 要查找的列名
    default: 任意值 - 如果找不到列时返回的默认值
    
    返回:
    str - 实际的列名，如果找不到则返回default值
    """
    # 直接匹配
    if col_name in df.columns:
        return col_name
    
    # 不区分大小写查找
    lower_col_name = col_name.lower()
    for col in df.columns:
        if col.lower() == lower_col_name:
            return col
    
    return default

def find_columns_case_insensitive(df, col_names):
    """
    以大小写不敏感方式从数据框中查找多个列名
    
    参数:
    df: DataFrame - 要查找列的数据框
    col_names: list - 要查找的列名列表
    
    返回:
    dict - 请求的列名到实际列名的映射，如果找不到则该列不包含在结果中
    """
    result = {}
    for name in col_names:
        actual_name = get_column_case_insensitive(df, name)
        if actual_name is not None:
            result[name] = actual_name
    return result

def normalize_column_case(df, lowercase=True):
    """
    标准化数据框列名的大小写
    
    参数:
    df: DataFrame - 要处理的数据框
    lowercase: bool - 为True时转为小写，False时转为大写
    
    返回:
    DataFrame - 列名已标准化的数据框副本
    """
    df_copy = df.copy()
    if lowercase:
        df_copy.columns = [col.lower() for col in df_copy.columns]
    else:
        df_copy.columns = [col.upper() for col in df_copy.columns]
    return df_copy

def get_df_with_columns_case_insensitive(df, columns):
    """
    返回数据框的子集，以大小写不敏感方式获取指定列
    
    参数:
    df: DataFrame - 要处理的数据框
    columns: list - 要获取的列名列表
    
    返回:
    DataFrame - 仅包含指定列的数据框
    """
    found_columns = []
    for col in columns:
        actual_col = get_column_case_insensitive(df, col)
        if actual_col is not None:
            found_columns.append(actual_col)
    
    if not found_columns:
        return pd.DataFrame()
    
    return df[found_columns]

def case_insensitive_column_exists(df, column_name):
    """
    检查列是否存在（不区分大小写）
    
    参数:
    df: DataFrame - 要检查的数据框
    column_name: str - 要检查的列名
    
    返回:
    bool - 如果列存在则为True，否则为False
    """
    return get_column_case_insensitive(df, column_name) is not None
