import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

# パワポ向けにアスペクト比を16:9に近づけ、左の図の割合を大きく
fig, axes = plt.subplots(2, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [2.2, 1], 'wspace': 0.15, 'hspace': 0.35})

np.random.seed(42)
theta = np.linspace(0, 2*np.pi, 6, endpoint=False)
x_pts = 2 + 0.9 * np.cos(theta)
y_pts = 5 + 0.9 * np.sin(theta)

text_bbox = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)

# --- 濃い色と大きなフォントサイズを定義 ---
c_y = '#08457e'   # 濃い青 (Previously c_x)
c_x1 = '#990000'  # 濃い赤 (Previously c_y1)
c_x2 = '#005a00'  # 濃い緑 (Previously c_y2)
c_line = '#555555'

title_fs = 24
ax_title_fs = 20
label_fs = 18
text_fs = 18
tick_fs = 14

pt_center = 200
pt_nbr = 80
arrow_lw = 1.5
mut_scale = 18

# ==========================================
# Row 1: H1 (Causal: neighbors -> neighbors)
# X -> Y (Left is Y, Right is X)
# ==========================================
ax_map1 = axes[0, 0]
ax_auc1 = axes[0, 1]

# MY space (Left)
circle_y1 = Circle((2, 5), 1.5, edgecolor=c_y, facecolor='#e6f2ff', lw=2.5)
ax_map1.add_patch(circle_y1)
ax_map1.scatter(2, 5, color=c_y, s=pt_center, zorder=6)
ax_map1.text(2, 5.3, '$Y(t)$', ha='center', va='bottom', fontsize=text_fs, fontweight='bold', color=c_y, bbox=text_bbox, zorder=7)
ax_map1.scatter(x_pts, y_pts, color=c_y, s=pt_nbr, zorder=5)
ax_map1.text(2, 3.2, '$Y^{kNN}(t)$', ha='center', va='top', fontsize=text_fs, fontweight='bold', color=c_y)

# MX space (Right)
r1, r2 = 1.0, 2.0
circle_x1_r2 = Circle((7, 5), r2, edgecolor=c_x1, facecolor='#fff2f2', lw=3, linestyle=':', alpha=0.7)
circle_x1_r1 = Circle((7, 5), r1, edgecolor=c_x1, facecolor='#ffe6e6', lw=2.5, linestyle='--', alpha=0.9)
ax_map1.add_patch(circle_x1_r2)
ax_map1.add_patch(circle_x1_r1)

# Mapped points
x_mapped_x = 7 + 0.5 * np.cos(theta + 0.3)
x_mapped_y = 5 + 0.5 * np.sin(theta + 0.3)
ax_map1.scatter(x_mapped_x, x_mapped_y, color=c_x1, s=pt_nbr, zorder=5, marker='s')

ax_map1.scatter(7, 5, color=c_x1, s=pt_center, zorder=6, marker='s')
ax_map1.text(7, 5.3, '$X(t)$', ha='center', va='bottom', fontsize=text_fs, fontweight='bold', color=c_x1, bbox=text_bbox, zorder=7)

# Radii arrows
angle_r1 = -np.pi/3
ax_map1.annotate('', xy=(7 + r1*np.cos(angle_r1), 5 + r1*np.sin(angle_r1)), 
                 xytext=(7, 5), arrowprops=dict(arrowstyle='-|>', color=c_x1, lw=2), zorder=4)
ax_map1.text(7 + (r1/2)*np.cos(angle_r1)+0.15, 5 + (r1/2)*np.sin(angle_r1)-0.1, '$r_1$', 
             color=c_x1, fontweight='bold', fontsize=text_fs, ha='center', va='center', bbox=text_bbox, zorder=7)

angle_r2 = -np.pi/7
ax_map1.annotate('', xy=(7 + r2*np.cos(angle_r2), 5 + r2*np.sin(angle_r2)), 
                 xytext=(7, 5), arrowprops=dict(arrowstyle='-|>', color=c_x1, lw=2), zorder=4)
ax_map1.text(7 + (r1 + (r2-r1)/2)*np.cos(angle_r2), 5 + (r1 + (r2-r1)/2)*np.sin(angle_r2)-0.15, '$r_2$', 
             color=c_x1, fontweight='bold', fontsize=text_fs, ha='center', va='center', bbox=text_bbox, zorder=7)

for i in range(6):
    arrow = FancyArrowPatch((x_pts[i], y_pts[i]), (x_mapped_x[i], x_mapped_y[i]),
                            arrowstyle='->', mutation_scale=mut_scale, color=c_line, lw=arrow_lw, alpha=0.8, zorder=3)
    ax_map1.add_patch(arrow)

# ズームインして図を大きく見せる
ax_map1.set_xlim(0, 9.5)
ax_map1.set_ylim(2.5, 7.5)
ax_map1.axis('off')
ax_map1.set_title(r'$H_1: X \Rightarrow Y$ (Causal)', fontsize=title_fs, fontweight='bold', pad=15)

# AUC Plot 1
r_vals = np.linspace(0, 1, 100)
ic_h1 = (1 - np.exp(-6 * r_vals)) / (1 - np.exp(-6))
ax_auc1.plot(r_vals, ic_h1, color=c_x1, lw=4)
ax_auc1.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
ax_auc1.fill_between(r_vals, ic_h1, color='#ffbb78', alpha=0.5)

r1_norm, r2_norm = 0.2, 0.5
y1_val = (1 - np.exp(-6 * r1_norm)) / (1 - np.exp(-6))
y2_val = (1 - np.exp(-6 * r2_norm)) / (1 - np.exp(-6))

ax_auc1.plot([r1_norm, r1_norm], [0, y1_val], color=c_x1, linestyle=':', lw=2.5)
ax_auc1.plot([r2_norm, r2_norm], [0, y2_val], color=c_x1, linestyle=':', lw=2.5)
ax_auc1.scatter([r1_norm, r2_norm], [y1_val, y2_val], color='white', edgecolor=c_x1, s=100, lw=3, zorder=5)

ticks = [0.0, r1_norm, 0.4, r2_norm, 0.6, 0.8, 1.0]
labels = ['0.0', '$r_1$', '0.4', '$r_2$', '0.6', '0.8', '1.0']
ax_auc1.set_xticks(ticks)
ax_auc1.set_xticklabels(labels, fontsize=tick_fs, fontweight='bold')
ax_auc1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_auc1.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=tick_fs, fontweight='bold')

for i, tick_label in enumerate(ax_auc1.get_xticklabels()):
    if i in [1, 3]:
        tick_label.set_color(c_x1)
        tick_label.set_fontsize(text_fs)

ax_auc1.set_xlim(0, 1)
ax_auc1.set_ylim(0, 1.05)
ax_auc1.set_xlabel('Threshold $r$', fontsize=label_fs, fontweight='bold')
ax_auc1.set_ylabel('Ratio', fontsize=label_fs, fontweight='bold')
ax_auc1.set_title(r'$AUC(H_1) > 0.5$', fontsize=ax_title_fs, fontweight='bold', color=c_x1)
ax_auc1.grid(True, linestyle=':', alpha=0.6)

# ==========================================
# Row 2: H0 (Non-causal: neighbors -> random)
# ==========================================
ax_map2 = axes[1, 0]
ax_auc2 = axes[1, 1]

# MY space
circle_y2 = Circle((2, 5), 1.5, edgecolor=c_y, facecolor='#e6f2ff', lw=2.5)
ax_map2.add_patch(circle_y2)
ax_map2.scatter(2, 5, color=c_y, s=pt_center, zorder=6)
ax_map2.text(2, 5.3, '$Y(t)$', ha='center', va='bottom', fontsize=text_fs, fontweight='bold', color=c_y, bbox=text_bbox, zorder=7)
ax_map2.scatter(x_pts, y_pts, color=c_y, s=pt_nbr, zorder=5)
ax_map2.text(2, 3.2, '$Y^{kNN}(t)$', ha='center', va='top', fontsize=text_fs, fontweight='bold', color=c_y)

# MX space
circle_x2_r2 = Circle((7, 5), r2, edgecolor=c_x2, facecolor='#f0fff0', lw=3, linestyle=':', alpha=0.7)
circle_x2_r1 = Circle((7, 5), r1, edgecolor=c_x2, facecolor='#e6ffe6', lw=2.5, linestyle='--', alpha=0.9)
ax_map2.add_patch(circle_x2_r2)
ax_map2.add_patch(circle_x2_r1)

# Mapped points
rand_radii = np.array([0.8, 1.8, 2.5, 2.8, 2.1, 1.4])
rand_angles = np.array([0.2, 1.1, 2.3, 3.5, 4.8, 5.7])
x_rand_x = 7 + rand_radii * np.cos(rand_angles)
x_rand_y = 5 + rand_radii * np.sin(rand_angles)
ax_map2.scatter(x_rand_x, x_rand_y, color=c_x2, s=pt_nbr, zorder=5, marker='s')

ax_map2.scatter(7, 5, color=c_x2, s=pt_center, zorder=6, marker='s')
ax_map2.text(7, 5.3, '$X(t)$', ha='center', va='bottom', fontsize=text_fs, fontweight='bold', color=c_x2, bbox=text_bbox, zorder=7)

# Radii arrows
ax_map2.annotate('', xy=(7 + r1*np.cos(angle_r1), 5 + r1*np.sin(angle_r1)), 
                 xytext=(7, 5), arrowprops=dict(arrowstyle='-|>', color=c_x2, lw=2), zorder=4)
ax_map2.text(7 + (r1/2)*np.cos(angle_r1)+0.15, 5 + (r1/2)*np.sin(angle_r1)-0.1, '$r_1$', 
             color=c_x2, fontweight='bold', fontsize=text_fs, ha='center', va='center', bbox=text_bbox, zorder=7)

ax_map2.annotate('', xy=(7 + r2*np.cos(angle_r2), 5 + r2*np.sin(angle_r2)), 
                 xytext=(7, 5), arrowprops=dict(arrowstyle='-|>', color=c_x2, lw=2), zorder=4)
ax_map2.text(7 + (r1 + (r2-r1)/2)*np.cos(angle_r2), 5 + (r1 + (r2-r1)/2)*np.sin(angle_r2)-0.15, '$r_2$', 
             color=c_x2, fontweight='bold', fontsize=text_fs, ha='center', va='center', bbox=text_bbox, zorder=7)

for i in range(6):
    arrow = FancyArrowPatch((x_pts[i], y_pts[i]), (x_rand_x[i], x_rand_y[i]),
                            arrowstyle='->', mutation_scale=mut_scale, color=c_line, lw=arrow_lw, alpha=0.8, zorder=3)
    ax_map2.add_patch(arrow)

# ズームイン
ax_map2.set_xlim(0, 9.5)
ax_map2.set_ylim(2.5, 7.5)
ax_map2.axis('off')
ax_map2.set_title(r'$H_0: X \nRightarrow Y$ (Non-causal)', fontsize=title_fs, fontweight='bold', pad=15)

# AUC Plot 2
ic_h0 = r_vals
ax_auc2.plot(r_vals, ic_h0, color=c_x2, lw=4)
ax_auc2.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
ax_auc2.fill_between(r_vals, ic_h0, color='#98df8a', alpha=0.5)

ax_auc2.plot([r1_norm, r1_norm], [0, r1_norm], color=c_x2, linestyle=':', lw=2.5)
ax_auc2.plot([r2_norm, r2_norm], [0, r2_norm], color=c_x2, linestyle=':', lw=2.5)
ax_auc2.scatter([r1_norm, r2_norm], [r1_norm, r2_norm], color='white', edgecolor=c_x2, s=100, lw=3, zorder=5)

ax_auc2.set_xticks(ticks)
ax_auc2.set_xticklabels(labels, fontsize=tick_fs, fontweight='bold')
ax_auc2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_auc2.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=tick_fs, fontweight='bold')
for i, tick_label in enumerate(ax_auc2.get_xticklabels()):
    if i in [1, 3]:
        tick_label.set_color(c_x2)
        tick_label.set_fontsize(text_fs)

ax_auc2.set_xlim(0, 1)
ax_auc2.set_ylim(0, 1.05)
ax_auc2.set_xlabel('Threshold $r$', fontsize=label_fs, fontweight='bold')
ax_auc2.set_ylabel('Ratio', fontsize=label_fs, fontweight='bold')
ax_auc2.set_title(r'$AUC(H_0) \approx 0.5$', fontsize=ax_title_fs, fontweight='bold', color=c_x2)
ax_auc2.grid(True, linestyle=':', alpha=0.6)

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    plt.tight_layout()
plt.savefig('cmc_schematic_ppt_swapped.png', dpi=300)