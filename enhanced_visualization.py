"""
增强的数据可视化模块
此模块提供了更高级的数据可视化功能，特别适用于论文中展示
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.gridspec as gridspec
from scipy.cluster import hierarchy
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

def create_correlation_heatmap(df, variables=None, output_path='output/figures/correlation_heatmap.png', 
                              method='pearson', cluster=True, mask_upper=False, annot=True):
    """
    创建相关系数热图
    
    参数:
    df: 包含变量的DataFrame
    variables: 要包含在热图中的变量列表，默认为None(使用所有数值型变量)
    output_path: 输出图像的路径
    method: 相关系数计算方法 ('pearson', 'spearman', 'kendall')
    cluster: 是否对变量进行聚类
    mask_upper: 是否隐藏上三角区域
    annot: 是否在热图中显示相关系数数值
    """
    try:
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 选择要用于热图的变量
        if variables is None:
            variables = df.select_dtypes(include=[np.number]).columns.tolist()
            
        # 确保所有选择的变量在DataFrame中存在
        valid_vars = [var for var in variables if var in df.columns]
        if not valid_vars:
            print("警告: 没有有效的数值型变量用于创建热图")
            return
        
        # 计算相关系数矩阵
        corr_matrix = df[valid_vars].corr(method=method)
        
        # 设置可视化参数
        plt.figure(figsize=(12, 10))
        
        # 创建掩码以隐藏上三角(如果需要)
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # 设置颜色映射
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        
        if cluster:
            # 基于相关系数进行聚类分析
            corr_linkage = hierarchy.ward(corr_matrix.values)
            
            # 创建聚类图
            g = sns.clustermap(
                corr_matrix, 
                cmap=cmap,
                mask=mask,
                annot=annot, 
                fmt=".2f",
                linewidths=0.5,
                figsize=(14, 12),
                row_linkage=corr_linkage,
                col_linkage=corr_linkage,
                vmin=-1, vmax=1,
                annot_kws={"size": 8}
            )
            
            # 旋转x轴标签
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
            plt.title(f"变量聚类相关系数矩阵 ({method}方法)", fontsize=16, pad=30)
        else:
            # 创建普通热图
            sns.heatmap(
                corr_matrix, 
                cmap=cmap,
                mask=mask,
                annot=annot, 
                fmt=".2f",
                linewidths=0.5,
                vmin=-1, vmax=1,
                annot_kws={"size": 8}
            )
            plt.title(f"变量相关系数矩阵 ({method}方法)", fontsize=16)
            
            # 旋转x轴标签
            plt.xticks(rotation=45, ha="right")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存相关系数热图到 {output_path}")
        
        # 保存高相关系数对到CSV文件
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # 只保留相关系数绝对值大于0.5的
                    high_corr_pairs.append({
                        'variable1': var1,
                        'variable2': var2,
                        'correlation': corr_value
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df = high_corr_df.sort_values('correlation', ascending=False)
            high_corr_csv = output_path.replace('.png', '_high_pairs.csv')
            high_corr_df.to_csv(high_corr_csv, index=False, encoding='utf-8-sig')
            print(f"已保存高相关系数对到 {high_corr_csv}")
            
        return corr_matrix
    
    except Exception as e:
        print(f"创建相关系数热图时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_pca_visualization(df, variables=None, n_components=2, 
                           output_path='output/figures/pca_visualization.png'):
    """
    进行主成分分析并创建可视化
    
    参数:
    df: 包含变量的DataFrame
    variables: 要包含在PCA中的变量列表，默认为None(使用所有数值型变量)
    n_components: 要提取的主成分数
    output_path: 输出图像的路径
    """
    try:
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 选择要用于PCA的变量
        if variables is None:
            variables = df.select_dtypes(include=[np.number]).columns.tolist()
            
        # 确保所有选择的变量在DataFrame中存在
        valid_vars = [var for var in variables if var in df.columns]
        if not valid_vars:
            print("警告: 没有有效的数值型变量用于主成分分析")
            return
            
        if len(valid_vars) < n_components:
            print(f"警告: 变量数量({len(valid_vars)})小于要提取的主成分数({n_components})，已调整为{len(valid_vars)}个主成分")
            n_components = len(valid_vars)
        
        # 准备数据
        X = df[valid_vars].dropna()
        if len(X) < 10:
            print("警告: 有效观测值太少，无法进行可靠的主成分分析")
            return
            
        # 标准化数据
        X_scaled = StandardScaler().fit_transform(X)
        
        # 进行PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X_scaled)
        
        # 创建主成分得分的DataFrame
        pc_df = pd.DataFrame(
            data=principal_components,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # 准备可视化
        plt.figure(figsize=(15, 12))
        
        # 设置子图布局
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5])
        
        # 1. 绘制解释方差比例
        ax1 = plt.subplot(gs[0, 0])
        explained_variance = pca.explained_variance_ratio_
        cum_explained_variance = np.cumsum(explained_variance)
        
        ax1.bar(range(1, n_components+1), explained_variance, alpha=0.7, color='skyblue')
        ax1.plot(range(1, n_components+1), cum_explained_variance, 'ro-')
        ax1.set_xlabel('主成分')
        ax1.set_ylabel('解释方差比例')
        ax1.set_title('主成分解释方差')
        ax1.grid(True, alpha=0.3)
        
        # 在每个柱子上显示解释方差比例
        for i, v in enumerate(explained_variance):
            ax1.text(i+1, v, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. 绘制变量与主成分的关系(载荷图)
        ax2 = plt.subplot(gs[0, 1])
        loadings = pca.components_
        
        # 处理和可视化载荷
        max_dims = min(5, n_components)  # 最多显示前5个主成分
        max_vars = min(10, len(valid_vars))  # 最多显示前10个变量
        
        # 选择载荷最高的变量
        abs_loadings = np.abs(loadings[:max_dims, :])
        sum_abs_loadings = np.sum(abs_loadings, axis=0)
        top_var_indices = np.argsort(-sum_abs_loadings)[:max_vars]
        
        # 创建热图
        sns.heatmap(
            loadings[:max_dims, top_var_indices],
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            xticklabels=[valid_vars[i] for i in top_var_indices],
            yticklabels=[f'PC{i+1}' for i in range(max_dims)],
            ax=ax2
        )
        ax2.set_title('主成分载荷')
        
        # 3. 如果有至少2个主成分，绘制PC1 vs PC2的散点图
        if n_components >= 2:
            ax3 = plt.subplot(gs[1, :])
            scatter = ax3.scatter(pc_df['PC1'], pc_df['PC2'], alpha=0.7)
            ax3.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
            ax3.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
            ax3.set_title('样本在PC1和PC2上的分布')
            ax3.grid(True, alpha=0.3)
            
            # 添加变量向量
            for i, var in enumerate(valid_vars):
                ax3.arrow(
                    0, 0,  # 从原点开始
                    pca.components_[0, i] * 7,  # 缩放以便适应图表
                    pca.components_[1, i] * 7,
                    head_width=0.1,
                    head_length=0.1,
                    width=0.01,
                    fc='red',
                    ec='red'
                )
                ax3.text(
                    pca.components_[0, i] * 7.5,
                    pca.components_[1, i] * 7.5,
                    var,
                    color='green',
                    ha='center',
                    va='center'
                )
            
            # 绘制原点十字线
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax3.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存PCA可视化到 {output_path}")
        
        # 保存PCA结果到CSV
        result_df = pd.DataFrame({
            'Principal_Component': [f'PC{i+1}' for i in range(n_components)],
            'Explained_Variance_Ratio': explained_variance,
            'Cumulative_Explained_Variance': cum_explained_variance
        })
        
        # 载荷矩阵
        loadings_df = pd.DataFrame(
            loadings,
            index=[f'PC{i+1}' for i in range(n_components)],
            columns=valid_vars
        )
        
        result_csv = output_path.replace('.png', '_results.csv')
        result_df.to_csv(result_csv, index=False, encoding='utf-8-sig')
        
        loadings_csv = output_path.replace('.png', '_loadings.csv')
        loadings_df.to_csv(loadings_csv, encoding='utf-8-sig')
        
        print(f"已保存PCA结果到 {result_csv} 和 {loadings_csv}")
        
        # 返回PCA对象和主成分得分
        return {'pca': pca, 'pc_scores': pc_df, 'loadings': loadings_df}
    
    except Exception as e:
        print(f"创建PCA可视化时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_industry_distribution(df, industry_col='indcd', 
                                output_path='output/figures/industry_distribution.png'):
    """
    创建行业分布分析图
    
    参数:
    df: 包含行业代码的DataFrame
    industry_col: 行业代码列名
    output_path: 输出图像的路径
    """
    try:
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 确保行业列存在
        if industry_col not in df.columns:
            print(f"警告: 行业列'{industry_col}'不在DataFrame中")
            return
        
        # 准备数据
        if isinstance(df.index, pd.MultiIndex):
            df_reset = df.reset_index()
        else:
            df_reset = df.copy()
        
        # 计算行业分布
        # 1. 按公司计算
        if 'stkcd' in df_reset.columns:
            # 确保每个公司只统计一次
            company_industry = df_reset.drop_duplicates(subset=['stkcd'])[industry_col]
            industry_counts = company_industry.value_counts().sort_values(ascending=False)
            
            if len(industry_counts) > 15:
                # 如果行业过多，只保留前10个，其余归为"其他"
                top_industries = industry_counts.head(10)
                others = pd.Series({'其他': industry_counts[10:].sum()})
                industry_counts = pd.concat([top_industries, others])
        else:
            # 否则直接计算行业列的频率
            industry_counts = df_reset[industry_col].value_counts().sort_values(ascending=False)
            if len(industry_counts) > 15:
                top_industries = industry_counts.head(10)
                others = pd.Series({'其他': industry_counts[10:].sum()})
                industry_counts = pd.concat([top_industries, others])
        
        # 创建行业分布图
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x=industry_counts.index, y=industry_counts.values, palette='viridis')
        
        # 添加数据标签
        for i, v in enumerate(industry_counts.values):
            ax.text(i, v + 0.5, f'{v}', ha='center')
        
        plt.title('样本公司行业分布')
        plt.xlabel('行业代码')
        plt.ylabel('公司数量')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存行业分布图到 {output_path}")
        
        # 创建饼图
        plt.figure(figsize=(12, 10))
        plt.pie(
            industry_counts.values, 
            labels=industry_counts.index,
            autopct='%1.1f%%', 
            startangle=90,
            shadow=True,
            explode=[0.05] + [0] * (len(industry_counts) - 1)  # 突出第一个行业
        )
        plt.axis('equal')
        plt.title('样本公司行业分布')
        
        pie_path = output_path.replace('.png', '_pie.png')
        plt.savefig(pie_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存行业分布饼图到 {pie_path}")
        
        # 保存行业分布数据
        industry_counts_df = pd.DataFrame({
            'industry': industry_counts.index,
            'count': industry_counts.values,
            'percentage': industry_counts.values / industry_counts.sum() * 100
        })
        
        csv_path = output_path.replace('.png', '.csv')
        industry_counts_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"已保存行业分布数据到 {csv_path}")
        
        return industry_counts_df
    
    except Exception as e:
        print(f"创建行业分布图时出错: {e}")
        import traceback
        traceback.print_exc()
        return None
        
def create_enhanced_descriptive_table(df, variables=None, include_diff=True, 
                                    output_path='output/tables/enhanced_descriptive_stats.xlsx'):
    """
    创建增强版描述性统计表，适合论文使用
    
    参数:
    df: 包含变量的DataFrame
    variables: 要包含在统计表中的变量列表，默认为None(使用所有数值型变量)
    include_diff: 是否包含以_diff结尾的差分变量
    output_path: 输出Excel文件的路径
    """
    try:
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 选择要用于统计表的变量
        if variables is None:
            variables = df.select_dtypes(include=[np.number]).columns.tolist()
            
        if not include_diff:
            variables = [var for var in variables if not var.endswith('_diff')]
            
        # 确保所有选择的变量在DataFrame中存在
        valid_vars = [var for var in variables if var in df.columns]
        if not valid_vars:
            print("警告: 没有有效的数值型变量用于创建描述性统计表")
            return
        
        # 计算描述性统计量
        desc_stats = df[valid_vars].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
        
        # 添加其他统计量
        desc_stats['Skewness'] = df[valid_vars].skew()
        desc_stats['Kurtosis'] = df[valid_vars].kurtosis()
        desc_stats['Missing (%)'] = df[valid_vars].isnull().mean() * 100
          # 计算变异系数(CV)，避免除零错误
        desc_stats['CV'] = desc_stats['std'] / desc_stats['mean'].replace(0, np.nan)
        
        # 数据格式优化
        formatted_stats = desc_stats.copy()
        
        # 进行格式化，保留合适的小数位数
        for col in formatted_stats.columns:
            if col == 'count':
                formatted_stats[col] = formatted_stats[col].astype(int)
            elif col == 'Missing (%)':
                formatted_stats[col] = formatted_stats[col].map('{:.2f}%'.format)
            else:
                formatted_stats[col] = formatted_stats[col].map('{:.4f}'.format)
        
        # 保存到Excel
        # 使用pandas ExcelWriter以获得更好的格式控制
        from pandas import ExcelWriter
        
        with ExcelWriter(output_path, engine='xlsxwriter') as writer:
            formatted_stats.to_excel(writer, sheet_name='描述性统计')
            
            # 获取工作簿和工作表对象
            workbook = writer.book
            worksheet = writer.sheets['描述性统计']
            
            # 定义格式
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            # 应用格式到标题行
            for col_num, value in enumerate(formatted_stats.columns.values):
                worksheet.write(0, col_num + 1, value, header_format)
                
            # 调整列宽
            worksheet.set_column(0, 0, 25)  # 第一列变量名
            worksheet.set_column(1, len(formatted_stats.columns), 12)  # 数据列
        
        print(f"已保存增强版描述性统计表到 {output_path}")
          # 生成简化版描述性统计表，适合直接插入论文
        # 使用 '50%' 替代 'median'，因为 describe 方法使用百分位数作为列名
        simple_cols = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'Skewness', 'Missing (%)']
        simple_stats = formatted_stats[simple_cols]
        
        # 重命名列，使其更适合论文
        simple_stats.columns = ['N', '均值', '标准差', '最小值', '25%分位', '中位数', '75%分位', '最大值', '偏度', '缺失率']
        
        # 保存简化版到CSV
        csv_path = output_path.replace('.xlsx', '_paper.csv')
        simple_stats.to_csv(csv_path, encoding='utf-8-sig')
        print(f"已保存论文友好版描述性统计表到 {csv_path}")
        
        return desc_stats
    
    except Exception as e:
        print(f"创建增强版描述性统计表时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def yearly_trends_analysis(df, key_vars=None, output_path='output/figures/yearly_trends_advanced.png'):
    """
    创建年度趋势高级分析图
    
    参数:
    df: 面板数据DataFrame
    key_vars: 要分析的关键变量列表
    output_path: 输出图像的路径
    """
    try:
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 重置多级索引
        if isinstance(df.index, pd.MultiIndex):
            df_reset = df.reset_index()
        else:
            df_reset = df.copy()
            
        # 确保'year'列存在
        if 'year' not in df_reset.columns:
            print("警告: 数据中不包含'year'列，无法进行年度趋势分析")
            return
            
        # 选择要分析的变量
        if key_vars is None:
            # 默认选择一些关键变量
            possible_vars = ['ai', 'ai_job_log', 'ai_patent_log', 
                           'intotal', 'ep', 'dp', 
                           'ai_patent_quality', 'ai_patent_depth']
            key_vars = [v for v in possible_vars if v in df_reset.columns]
            
        if not key_vars:
            print("警告: 没有有效的变量用于年度趋势分析")
            return
            
        # 限制变量数量以避免图表过于拥挤
        if len(key_vars) > 6:
            print(f"警告: 变量过多，仅使用前6个变量进行趋势分析: {key_vars[:6]}")
            key_vars = key_vars[:6]
        
        # 计算年度均值和标准差
        yearly_means = df_reset.groupby('year')[key_vars].mean()
        yearly_stds = df_reset.groupby('year')[key_vars].std()
        
        # 创建高级趋势图
        fig, axes = plt.subplots(len(key_vars), 1, figsize=(12, 4 * len(key_vars)))
        if len(key_vars) == 1:
            axes = [axes]
            
        colors = plt.cm.tab10(np.linspace(0, 1, len(key_vars)))
        
        for i, var in enumerate(key_vars):
            ax = axes[i]
            mean_values = yearly_means[var]
            std_values = yearly_stds[var]
            
            # 绘制均值线
            ax.plot(yearly_means.index, mean_values, 'o-', color=colors[i], linewidth=2, label='均值')
            
            # 绘制标准差区域
            ax.fill_between(
                yearly_means.index, 
                mean_values - std_values, 
                mean_values + std_values,
                alpha=0.2,
                color=colors[i],
                label='±标准差'
            )
            
            # 计算趋势线和斜率
            try:
                years = yearly_means.index.astype(float).values
                X = sm.add_constant(years)
                model = sm.OLS(mean_values.values, X).fit()
                trend_line = model.predict(X)
                slope = model.params[1]
                p_value = model.pvalues[1]
                
                # 绘制趋势线
                ax.plot(yearly_means.index, trend_line, '--', color='red', 
                       label=f'趋势线 (斜率={slope:.4f}, p={p_value:.4f})')
                
                # 添加均值点标签
                for year, value in zip(yearly_means.index, mean_values):
                    ax.text(year, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
                
            except Exception as trend_err:
                print(f"计算{var}的趋势线时出错: {trend_err}")
            
            ax.set_title(f'{var}的年度变化趋势', fontsize=12)
            ax.set_xlabel('年份')
            ax.set_ylabel('数值')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 添加年度总结
            if len(yearly_means.index) > 2:
                first_year = yearly_means.index[0]
                last_year = yearly_means.index[-1]
                first_val = mean_values.iloc[0]
                last_val = mean_values.iloc[-1]
                change_pct = (last_val - first_val) / abs(first_val) * 100 if first_val != 0 else np.inf
                
                change_text = f"从{first_year}到{last_year}，{var}增长了{change_pct:.1f}%"
                if change_pct < 0:
                    change_text = f"从{first_year}到{last_year}，{var}下降了{abs(change_pct):.1f}%"
                    
                ax.text(0.5, 0.01, change_text, transform=ax.transAxes, ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存年度趋势分析图到 {output_path}")
        
        # 创建合并趋势图
        plt.figure(figsize=(14, 8))
        
        for i, var in enumerate(key_vars):
            # 对每个变量进行标准化，以便于在同一图表上比较
            mean_values = yearly_means[var]
            normalized_values = (mean_values - mean_values.min()) / (mean_values.max() - mean_values.min())
            plt.plot(yearly_means.index, normalized_values, 'o-', linewidth=2, label=var)
        
        plt.title('关键变量年度变化趋势比较(标准化)', fontsize=14)
        plt.xlabel('年份', fontsize=12)
        plt.ylabel('标准化数值', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        combined_path = output_path.replace('.png', '_combined.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存合并年度趋势图到 {combined_path}")
        
        # 保存年度趋势数据
        trend_data = pd.DataFrame(index=yearly_means.index)
        
        for var in key_vars:
            trend_data[f'{var}_mean'] = yearly_means[var]
            trend_data[f'{var}_std'] = yearly_stds[var]
        
        csv_path = output_path.replace('.png', '.csv')
        trend_data.to_csv(csv_path, encoding='utf-8-sig')
        print(f"已保存年度趋势数据到 {csv_path}")
        
        return {'means': yearly_means, 'stds': yearly_stds, 'trend_data': trend_data}
    
    except Exception as e:
        print(f"创建年度趋势分析图时出错: {e}")
        import traceback
        traceback.print_exc()
        return None
