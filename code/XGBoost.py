import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import warnings
from scipy.stats import zscore, pearsonr
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import scipy.stats as ss
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.model_selection import GridSearchCV
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 忽略警告
warnings.filterwarnings('ignore')
# 1. 数据加载与预处理
print("=== 步骤1: 数据加载与预处理 ===")
df = pd.read_csv('10.9+HQ改.csv')
target = 'ACL stress'
#drop_cols = ['ACL stress']
drop_cols = ['ACL stress']
X = df.drop(drop_cols, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 2. 全特征初步调参
print("\n=== 步骤2: 全特征参数优化 ===")
def visualize_param_tuning(param_name, param_range, fixed_params):
    """可视化参数调优"""
    rs = []  # R²均值
    var = [] # R²方差
    ge = []  # 泛化误差

    for value in param_range:
        params = fixed_params.copy()
        params[param_name] = value
        model = xgb.XGBRegressor(**params)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        rs.append(cv_scores.mean())
        var.append(cv_scores.var())
        ge.append((1 - cv_scores.mean())**2 + cv_scores.var())

    best_r2_idx = np.argmax(rs)
    best_var_idx = np.argmin(var)
    best_ge_idx = np.argmin(ge)

    print(f"\n{param_name}优化结果:")
    print(f"最佳R²: {param_name}={param_range[best_r2_idx]}, R²={rs[best_r2_idx]:.4f}, 方差={var[best_r2_idx]:.6f}")
    print(f"最小泛化误差: {param_name}={param_range[best_ge_idx]}, 泛化误差={ge[best_ge_idx]:.6f}")

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(param_range, rs, 'o-', color='royalblue')
    plt.axvline(param_range[best_r2_idx], color='red', linestyle='--')
    plt.title(f'{param_name} vs R²')
    plt.grid(True, alpha=0.3)

    plt.subplot(132)
    plt.plot(param_range, var, 'o-', color='darkorange')
    plt.axvline(param_range[best_var_idx], color='red', linestyle='--')
    plt.title(f'{param_name} vs 方差')
    plt.grid(True, alpha=0.3)

    plt.subplot(133)
    plt.plot(param_range, ge, 'o-', color='forestgreen')
    plt.axvline(param_range[best_ge_idx], color='red', linestyle='--')
    plt.title(f'{param_name} vs 泛化误差')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return param_range[best_ge_idx]

# 固定参数
fixed_params = {
    'learning_rate': 0.05,
    'max_depth': 2,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'random_state': 42
}

print("\n优化n_estimators...")
fixed_params['n_estimators'] = visualize_param_tuning('n_estimators', range(50, 501, 50), fixed_params)

print("\n优化learning_rate...")
fixed_params['learning_rate'] = visualize_param_tuning('learning_rate', [0.001, 0.005, 0.01, 0.05, 0.1, 0.2], fixed_params)

print("\n优化max_depth...")
fixed_params['max_depth'] = visualize_param_tuning('max_depth', [2, 3, 4, 5, 6], fixed_params)
# === 步骤 3: 特征重要性分析（原步骤 4） ===
model = xgb.XGBRegressor(**fixed_params)
model.fit(X_train, y_train)

importance = pd.Series(model.feature_importances_, index=X.columns)
important_features = importance[importance > 0.01].index.tolist()  # 修改为英文变量名

plt.figure(figsize=(10, 6))
importance.sort_values().plot.barh()
plt.title("特征重要性（全特征空间）")
plt.xlabel("重要性分数")
plt.ylabel("特征")
plt.grid(True)
plt.show()
print(f"\n重要特征数量: {len(important_features)}")
print("重要特征列表:", important_features)
print("\n=== 步骤3.5: 重要特征子集核心参数重新调优 ===")

# 使用步骤3筛选的重要特征
X_train_sel = X_train[important_features]

# 继承全特征调优的参数（或自定义初始值）
refined_params = {
    'learning_rate': fixed_params['learning_rate'],  # 继承原学习率
    'max_depth': fixed_params['max_depth'],         # 继承原深度
    'n_estimators': fixed_params['n_estimators'],   # 继承原树数量
    'reg_alpha': fixed_params['reg_alpha'],         # 其他参数保持不变
    'reg_lambda': fixed_params['reg_lambda'],
    'subsample': fixed_params['subsample'],
    'colsample_bytree': fixed_params['colsample_bytree'],
    'random_state': 42
}

# ---- 重新优化n_estimators（范围可调整）----
print("\n重新优化 n_estimators...")
refined_params['n_estimators'] = visualize_param_tuning(
    param_name='n_estimators',
    param_range=range(50, 401, 50),  # 特征减少后可能需要更少的树
    fixed_params=refined_params
)

# ---- 重新优化learning_rate（范围可调整）----
print("\n重新优化 learning_rate...")
refined_params['learning_rate'] = visualize_param_tuning(
    param_name='learning_rate',
    param_range=[0.001, 0.01, 0.05, 0.1, 0.2],  # 更精细的搜索
    fixed_params=refined_params
)

# ---- 重新优化max_depth（范围可调整）----
print("\n重新优化 max_depth...")
refined_params['max_depth'] = visualize_param_tuning(
    param_name='max_depth',
    param_range=[2, 3, 4, 5],  # 特征减少后可能不需要太深的树
    fixed_params=refined_params
)

# 输出最终调优结果
print("\n⭐️ 重要特征子集最终核心参数:")
for k, v in refined_params.items():
    print(f"{k}: {v}")

# 后续网格搜索将使用 refined_params 替代原 fixed_params
# 4. 网格搜索优化
print("\n=== 步骤4: 网格搜索优化 ===")
param_grid = {
    'reg_alpha': [0.5, 1.0, 2.0],
    'reg_lambda': [1.0, 2.0, 5.0],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'gamma': [0, 0.1, 0.2]
}

X_train_sel = X_train[important_features]
X_test_sel = X_test[important_features]

grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(**fixed_params),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train_sel, y_train)
print("\n最优参数:", grid_search.best_params_)
print("最优R²分数:", grid_search.best_score_)
# 5. 最终模型评估
print("\n=== 步骤5: 最终模型评估 ===")
final_params = fixed_params.copy()
final_params.update(grid_search.best_params_)
final_model = xgb.XGBRegressor(**final_params)
final_model.fit(X_train_sel, y_train)

train_pred = final_model.predict(X_train_sel)
test_pred = final_model.predict(X_test_sel)

print("\n最终模型性能:")
print(f"训练集R²: {r2_score(y_train, train_pred):.4f}")
print(f"测试集R²: {r2_score(y_test, test_pred):.4f}")
print(f"泛化gap: {r2_score(y_train, train_pred) - r2_score(y_test, test_pred):.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, test_pred, color='dodgerblue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("实际值ACL stress")
plt.ylabel("预测值ACL stress")
plt.title("最终模型: 实际值 vs 预测值")
plt.grid(True)
plt.show()
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 计算预测值
y_pred = model.predict(X_test)

# 计算并打印回归指标
print("\n回归性能指标:")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
import matplotlib
import matplotlib.pyplot as plt

# 设置中文字体，避免中文乱码
matplotlib.rcParams['font.family'] = 'SimHei'   # 黑体（支持中文）
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 5. 最终模型评估
print("\n=== 步骤5: 最终模型评估 ===")
final_params = fixed_params.copy()
final_params.update(grid_search.best_params_)
final_model = xgb.XGBRegressor(**final_params)
final_model.fit(X_train_sel, y_train)

# 预测结果
train_pred = final_model.predict(X_train_sel)
test_pred = final_model.predict(X_test_sel)

# 计算所有评估指标
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
train_rmse = mean_squared_error(y_train, train_pred, squared=False)
test_rmse = mean_squared_error(y_test, test_pred, squared=False)
train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

# 格式化输出评估结果
print("\n最终模型性能:")
print(f"{'指标':<10}{'训练集':<15}{'测试集':<15}{'泛化gap':<10}")
print(f"{'R²':<10}{train_r2:.4f}{'':<5}{test_r2:.4f}{'':<5}{train_r2 - test_r2:.4f}")
print(f"{'RMSE':<10}{train_rmse:.4f}{'':<5}{test_rmse:.4f}{'':<5}{train_rmse - test_rmse:.4f}")
print(f"{'MAE':<10}{train_mae:.4f}{'':<5}{test_mae:.4f}{'':<5}{train_mae - test_mae:.4f}")

# ================= 可视化 =================
plt.figure(figsize=(6, 5))  # 图更小
plt.scatter(y_test, test_pred, color='dodgerblue', alpha=0.7, label="Predicted points")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label="Ideal diagonal (y=x)")

# 调整字体大小（中文不会乱码）
plt.xlabel("Actual ACL stress", fontsize=14)
plt.ylabel("Predicted ACL stress", fontsize=14)


# 坐标刻度字体
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 添加文本框标注 R²、RMSE、MAE（R² 用 LaTeX）
textstr = '\n'.join((
    fr"$R^2$ = {test_r2:.3f}",
    f"RMSE = {test_rmse:.3f}",
    f"MAE = {test_mae:.3f}"
))
plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.legend(fontsize=12)
plt.grid(True, alpha=0.5)
plt.tight_layout()

# 保存图片
plt.savefig("最终模型评估.png", dpi=300, bbox_inches="tight")
plt.show()
import joblib
import os

# ================= 保存模型 =================
model_filename = "final_XGJ_model.bin"
joblib.dump(final_model, model_filename)

print("\n=== 模型已成功保存 ===")
print(f"保存路径: {os.path.abspath(model_filename)}")
import os
import shap
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

# ===== Global Font Settings =====
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = True
matplotlib.rcParams['pdf.fonttype'] = 42

# ================= Initial Setup =================
save_dir = f"SHAP_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)

# ================= SHAP Calculation =================
explainer = shap.Explainer(final_model, X_train_sel)
shap_values = explainer(X_test_sel)
shap_array = shap_values.values

# ================= Basic Plots =================
def plot_basic_shap():
    """Generate basic SHAP plots"""
    # 1. Bar plot
    plt.figure(figsize=(10, max(6, len(X_test_sel.columns)*0.3)))
    shap.plots.bar(shap_values, max_display=len(X_test_sel.columns), show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_feature_importance_all.png'), dpi=300)
    plt.close()

    # 2. Beeswarm plot
    plt.figure(figsize=(10, max(6, len(X_test_sel.columns)*0.4)))
    shap.plots.beeswarm(shap_values, max_display=len(X_test_sel.columns), show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_beeswarm_all.png'), dpi=300)
    plt.close()

    # 3. Violin plot
    plt.figure(figsize=(12, max(6, len(X_test_sel.columns)*0.4)))
    shap.plots.violin(shap_values, max_display=len(X_test_sel.columns), show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3_violin_all.png'), dpi=300)
    plt.close()

# ================= Advanced Analysis =================
def plot_advanced_shap():
    """Generate advanced analysis plots"""
    try:
        # 4. Decision plot
        custom_sample_indices = [52, 0, 12, 14, 30]
        plt.figure(figsize=(12, 10))
        shap.decision_plot(
            explainer.expected_value,
            shap_array[custom_sample_indices],
            feature_names=X_test_sel.columns.tolist(),
            highlight=0,
            legend_labels=[f'sample {i+1}' for i in range(5)],
            show=False,
            feature_order='importance'
        )
        plt.tight_layout(pad=3)
        plt.savefig(os.path.join(save_dir, '4_decision_plot.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"[Warning] Decision plot generation failed: {str(e)}")

    try:
        # 5. Heatmap (safe version with bounds checking)
        plot_samples = min(50, len(shap_values))  # Ensure we don't exceed array bounds
        plt.figure(figsize=(14, 10))
        shap.plots.heatmap(
            shap_values[:plot_samples],
            instance_order=np.argsort(shap_array[:plot_samples].sum(1)),
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '5_heatmap.png'), dpi=300)
        plt.close()
        print(f"Generated heatmap for first {plot_samples} samples")
    except Exception as e:
        print(f"[Warning] Heatmap generation failed: {str(e)}")

    try:
        # 6. Interaction plot
        if all(col in X_test_sel.columns for col in ['knee_flexion', 'knee_valgus']):
            plt.figure(figsize=(12, 7))
            shap.plots.scatter(
                shap_values[:, 'knee_flexion'],
                color=shap_values[:, 'knee_valgus'],
                show=False
            )
            plt.title("Knee Flexion-Valgus Interaction", fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, '6_interaction_plot.png'), dpi=300)
            plt.close()
    except Exception as e:
        print(f"[Warning] Interaction plot generation failed: {str(e)}")

# ================= Individual Explanations =================
def plot_individual_shap():
    """Generate individual sample explanation plots with proper number formatting"""
    sample_idx = min(52, len(shap_values)-1)  # Ensure sample index is within bounds
    
    # ===== 5. Waterfall Plot =====
    try:
        print("\n5. Individual Prediction Breakdown - Waterfall Plot (PNG)")
        
        # Custom number formatting for waterfall plot
        def format_value(val, format_str=None):
            rounded_val = round(val, 2)
            if abs(rounded_val) < 0.005:  # If would display as 0.00
                return "0.01" if val >= 0 else "-0.01"
            elif rounded_val == int(rounded_val):  # Integer values
                return f"{int(rounded_val)}"
            else:                # Float values
                return f"{rounded_val:.2f}"
        
        # Monkey patch the formatter (temporary solution)
        original_format = shap.plots._waterfall.format_value
        shap.plots._waterfall.format_value = format_value
        
        plt.figure(figsize=(14, 8))
        shap.plots.waterfall(
            shap_values[sample_idx],
            max_display=20,
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '8_waterfall_plot.png'), dpi=300)
        plt.close()
        
        # Restore original formatter
        shap.plots._waterfall.format_value = original_format
        
        print(f"Successfully generated waterfall plot for sample {sample_idx}")
    except Exception as e:
        print(f"[Warning] Waterfall plot generation failed: {str(e)}")

    # ===== 6. Force Plot (Interactive HTML) =====
    try:
        print("\n6. Interactive Force Plot (hover to see feature values)")
        
        # Prepare feature values (rounded to 2 decimals)
        X_force = X_test_sel.iloc[sample_idx].copy()
        
        # Apply rounding and replace 0.00 with 0.01
        def safe_round(x):
            rounded = round(x, 2)
            if abs(rounded) < 0.005:  # Threshold for considering as zero
                return 0.01 if x >= 0 else -0.01
            return rounded
            
        X_force_rounded = X_force.apply(safe_round)
        
        # Create Explanation object for the single sample
        explanation = shap.Explanation(
            values=shap_values[sample_idx].values,
            base_values=explainer.expected_value,
            data=X_force_rounded.values,
            feature_names=X_test_sel.columns.tolist()
        )
        
        # Create and save force plot
        force_plot = shap.plots.force(explanation, matplotlib=False)
        shap.save_html(
            os.path.join(save_dir, '9_force_plot.html'), 
            force_plot
        )
        print(f"Successfully generated force plot for sample {sample_idx}")
    except Exception as e:
        print(f"[Warning] HTML force plot generation failed: {str(e)}")

# ================= Execute All Plot Generation =================
plot_basic_shap()
plot_advanced_shap()
plot_individual_shap()

# ================= Data Output =================
shap_df = pd.DataFrame({
    'Feature': X_test_sel.columns,
    'Mean|SHAP|': np.abs(shap_array).mean(axis=0).round(4),
    'Direction': np.where(shap_array.mean(axis=0) > 0, 'Positive', 'Negative')
}).sort_values('Mean|SHAP|', ascending=False)

shap_df.to_csv(os.path.join(save_dir, 'feature_importance_full.csv'), index=False)

print(f"\nAnalysis complete! Results saved to: {os.path.abspath(save_dir)}")
print("Generated files:")
print('\n'.join([f"• {f}" for f in os.listdir(save_dir)]))
import os
import numpy as np
import matplotlib.pyplot as plt
import shap
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ===== 初始化设置 =====
save_dir = f"SHAP紧凑图_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)

# ===== 特征名称中英映射 =====
FEATURE_NAME_MAP = {
    'HFA': 'HFA(°)',
    'KFA': 'KFA(°)' ,
    'AFA': 'AFA(°)',
    'HAA': 'HAA(°)',
    'KVA': 'KVA(°)',
    'AVA': 'AVA(°)',
    'KVM': 'KVM(Nm/kg)',
    'KFM': 'KFM(Nm/kg)',
    'ASF': 'ASF(N/BW)',
 }

# ===== 辅助函数 =====
def sanitize_feature_name(feature_name):
    """将特征名中的非法字符替换为下划线"""
    return feature_name.replace('/', '_').replace('\\', '_')

def translate_feature_names(df):
    """将数据框的列名翻译为中文"""
    return df.rename(columns=lambda x: FEATURE_NAME_MAP.get(x, x))

# ===== 预处理特征名称 =====
X_train_sel = translate_feature_names(X_train_sel)
X_test_sel = translate_feature_names(X_test_sel)

# ===== SHAP值计算 =====
explainer = shap.Explainer(final_model, X_train_sel)
shap_values = explainer(X_test_sel)
shap_array = shap_values.values  # SHAP值数组

# ===== 超迷你单图（颜色条等长） =====
def plot_compact_dependence():
    """生成适合论文的超迷你SHAP依赖图，颜色条等长"""
    for feature in X_test_sel.columns:
        try:
            fig, ax = plt.subplots(figsize=(2.4, 1.8))  # 小尺寸画布     
            sc = ax.scatter(
                X_test_sel[feature],
                shap_array[:, X_test_sel.columns.get_loc(feature)],
                c=shap_array[:, X_test_sel.columns.get_loc(feature)],
                cmap=shap.plots.colors.red_blue,
                vmin=-np.abs(shap_array[:, X_test_sel.columns.get_loc(feature)]).max(),
                vmax=np.abs(shap_array[:, X_test_sel.columns.get_loc(feature)]).max(),
                s=7,
                alpha=0.8,
                linewidths=0.2
            )

            # 创建等长颜色条
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(sc, cax=cax)
            cbar.set_label('SHAP值', rotation=270, labelpad=5, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            
            ax.set_xlabel(feature, fontsize=9, labelpad=0.5)
            ax.set_ylabel("SHAP value", fontsize=9, labelpad=0.5)
            ax.spines[['top', 'right', 'left', 'bottom']].set_linewidth(0.4)
            ax.tick_params(axis='both', which='major', labelsize=7, width=0.4)

            plt.tight_layout(pad=0.15)
            
            # 处理文件名中的非法字符
            safe_feature = sanitize_feature_name(feature)
            plt.savefig(
                os.path.join(save_dir, f'{safe_feature}_紧凑图.png'),
                dpi=450,
                bbox_inches='tight',
                facecolor='white'
            )
            plt.close()
            print(f"✓ {feature} (已保存为'{safe_feature}_紧凑图.png')")
        except Exception as e:
            print(f"✗ {feature}: {str(e)}")

# ===== 拼图版本（3行4列，每个子图保持自己的颜色范围） =====
def combine_plots_to_one():
    """将所有SHAP依赖图拼成3行4列的论文图"""
    feature_list = list(X_test_sel.columns)
    cols = 3  # 固定4列
    rows = 3   # 固定3行
    
    # 检查特征数量是否超过子图数量
    if len(feature_list) > rows * cols:
        print(f"警告: 特征数量({len(feature_list)})超过子图数量({rows*cols})，将只显示前{rows*cols}个特征")
        feature_list = feature_list[:rows*cols]

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.4, rows*1.8), sharex=False, sharey=False)
    
    # 如果只有一行，确保axes是二维数组
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # 存储所有scatter对象用于颜色条
    scatter_objects = []

    for i, feature in enumerate(feature_list):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        feature_shap = shap_array[:, X_test_sel.columns.get_loc(feature)]
        abs_max = np.abs(feature_shap).max()
        
        sc = ax.scatter(
            X_test_sel[feature],
            feature_shap,
            c=feature_shap,
            cmap=shap.plots.colors.red_blue,
            vmin=-abs_max,
            vmax=abs_max,
            s=7,
            alpha=0.8,
            linewidths=0.2
        )
        scatter_objects.append(sc)
        
        ax.set_xlabel(feature, fontsize=9, labelpad=0.5)
        ax.set_ylabel("SHAP value", fontsize=9, labelpad=0.5)
        ax.spines[['top', 'right', 'left', 'bottom']].set_linewidth(0.4)
        ax.tick_params(axis='both', which='major', labelsize=7, width=0.4)

    # 删除多余子图
    for i in range(len(feature_list), rows*cols):
        row = i // cols
        col = i % cols
        fig.delaxes(axes[row, col])

    # 添加公共颜色条（使用最后一个scatter对象）
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(scatter_objects[-1], cax=cbar_ax)
    cbar.set_label('SHAP value', rotation=270, labelpad=5, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout(pad=0.3, rect=[0, 0, 0.9, 1])  # 为颜色条留空间
    plt.savefig(
        os.path.join(save_dir, 'SHAP_3x4_紧凑图.png'),
        dpi=450,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close()
    print(f"✓ 3x4多图组合已保存至 {save_dir}/SHAP_3x4_紧凑图.png")

# ===== 执行 =====
print("正在生成超迷你SHAP依赖图...")
plot_compact_dependence()

print("\n正在生成3x4多图组合...")
combine_plots_to_one()

print(f"\n所有紧凑图已保存至: {os.path.abspath(save_dir)}")

print(f"已处理特征总数: {len(X_test_sel.columns)}")

