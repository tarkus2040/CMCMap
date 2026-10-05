import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

fig, axes = plt.subplots(2, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [2.2, 1], 'wspace': 0.15, 'hspace': 0.35})

np.random.seed(42)

# Original 6 Y points
theta = np.linspace(0, 2*np.pi, 6, endpoint=False)
x_pts_orig = 2 + 0.9 * np.cos(theta)
y_pts_orig = 5 + 0.9 * np.sin(theta)

# 2 additional Y points for the boundary mappings
angle_y_add1 = 1.5 
angle_y_add2 = 4.7
y_add_x = np.array([2 + 0.9 * np.cos(angle_y_add1), 2 + 0.9 * np.cos(angle_y_add2)])
y_add_y = np.array([5 + 0.9 * np.sin(angle_y_add1), 5 + 0.9 * np.sin(angle_y_add2)])

# Combine to 8 points total in Y space
all_y_x = np.concatenate([x_pts_orig, y_add_x])
all_y_y = np.concatenate([y_pts_orig, y_add_y])

text_bbox = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)

c_y = '#08457e'   # Deep Blue
c_x1 = '#990000'  # Deep Red
c_x2 = '#005a00'  # Deep Green
c_line = '#555555'

title_fs = 24
ax_title_fs = 20
label_fs = 18
text_fs = 18
tick_fs = 14
manifold_fs = 22

pt_center = 200
pt_nbr = 80
arrow_lw = 1.5
mut_scale = 18

r1, r2 = 1.0, 2.0
angle_pt1 = np.pi / 4
angle_pt2 = 3 * np.pi / 4

# ==========================================
# Row 1: H1 (Causal: neighbors -> neighbors)
# ==========================================
ax_map1 = axes[0, 0]
ax_auc1 = axes[0, 1]

# Draw a divider and background shades to clearly separate the two state spaces
ax_map1.axvspan(-1, 4.5, facecolor='#f4f7f9', alpha=0.6, zorder=0)  # Subtle background for MY
ax_map1.axvspan(4.5, 10, facecolor='#f9f4f4', alpha=0.6, zorder=0)  # Subtle background for MX
ax_map1.axvline(x=4.5, color='gray', linestyle='-.', lw=2, alpha=0.5, zorder=1)

# MY space (Left)
circle_y1 = Circle((2, 5), 1.5, edgecolor=c_y, facecolor='#e6f2ff', lw=2.5)
ax_map1.add_patch(circle_y1)
ax_map1.scatter(2, 5, color=c_y, s=pt_center, zorder=6)
ax_map1.text(2, 5.3, '$Y(t)$', ha='center', va='bottom', fontsize=text_fs, fontweight='bold', color=c_y, bbox=text_bbox, zorder=7)
ax_map1.scatter(all_y_x, all_y_y, color=c_y, s=pt_nbr, zorder=5)
ax_map1.text(2, 3.1, '$M_Y$', ha='center', va='top', fontsize=manifold_fs, fontweight='bold', color=c_y)

# MX space (Right)
circle_x1_r2 = Circle((7, 5), r2, edgecolor=c_x1, facecolor='#fff2f2', lw=3, linestyle=':', alpha=0.7)
circle_x1_r1 = Circle((7, 5), r1, edgecolor=c_x1, facecolor='#ffe6e6', lw=2.5, linestyle='--', alpha=0.9)
ax_map1.add_patch(circle_x1_r2)
ax_map1.add_patch(circle_x1_r1)
ax_map1.text(7, 2.5, '$M_X$', ha='center', va='top', fontsize=manifold_fs, fontweight='bold', color=c_x1)

# Mapped points
x_mapped_orig_x = 7 + 0.5 * np.cos(theta + 0.3)
x_mapped_orig_y = 5 + 0.5 * np.sin(theta + 0.3)

# 2 additional boundary points in X space for H1
x_add_x1 = 7 + r1 * np.cos(angle_pt1)
x_add_y1 = 5 + r1 * np.sin(angle_pt1)
x_add_x2 = 7 + r2 * np.cos(angle_pt2)
x_add_y2 = 5 + r2 * np.sin(angle_pt2)

all_x1_x = np.concatenate([x_mapped_orig_x, [x_add_x1, x_add_x2]])
all_x1_y = np.concatenate([x_mapped_orig_y, [x_add_y1, x_add_y2]])

ax_map1.scatter(all_x1_x, all_x1_y, color=c_x1, s=pt_nbr, zorder=5, marker='s')

ax_map1.scatter(7, 5, color=c_x1, s=pt_center, zorder=6, marker='s')
ax_map1.text(7, 5.3, '$X(t)$', ha='center', va='bottom', fontsize=text_fs, fontweight='bold', color=c_x1, bbox=text_bbox, zorder=7)

# Radii arrows
angle_r1 = -np.pi / 3
ax_map1.annotate('', xy=(7 + r1 * np.cos(angle_r1), 5 + r1 * np.sin(angle_r1)), 
                 xytext=(7, 5), arrowprops=dict(arrowstyle='-|>', color=c_x1, lw=2), zorder=4)
ax_map1.text(7 + (r1 / 2) * np.cos(angle_r1) + 0.15, 5 + (r1 / 2) * np.sin(angle_r1) - 0.1, '$r_1$', 
             color=c_x1, fontweight='bold', fontsize=text_fs, ha='center', va='center', bbox=text_bbox, zorder=7)

angle_r2 = -np.pi / 7
ax_map1.annotate('', xy=(7 + r2 * np.cos(angle_r2), 5 + r2 * np.sin(angle_r2)), 
                 xytext=(7, 5), arrowprops=dict(arrowstyle='-|>', color=c_x1, lw=2), zorder=4)
ax_map1.text(7 + (r1 + (r2 - r1) / 2) * np.cos(angle_r2), 5 + (r1 + (r2 - r1) / 2) * np.sin(angle_r2) - 0.15, '$r_2$', 
             color=c_x1, fontweight='bold', fontsize=text_fs, ha='center', va='center', bbox=text_bbox, zorder=7)

# Draw arrows for all 8 points
for i in range(len(all_y_x)):
    arrow = FancyArrowPatch((all_y_x[i], all_y_y[i]), (all_x1_x[i], all_x1_y[i]),
                            arrowstyle='->', mutation_scale=mut_scale, color=c_line, lw=arrow_lw, alpha=0.8, zorder=3)
    ax_map1.add_patch(arrow)

ax_map1.set_xlim(0, 9.5)
ax_map1.set_ylim(2.2, 7.5)
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
ax_auc1.set_xlabel('kNN Rank Fraction ($r$)', fontsize=label_fs, fontweight='bold')
ax_auc1.set_ylabel('True Positive Rate', fontsize=label_fs, fontweight='bold')
ax_auc1.set_title(r'$AUC(H_1) > 0.5$', fontsize=ax_title_fs, fontweight='bold', color=c_x1)
ax_auc1.grid(True, linestyle=':', alpha=0.6)

# ==========================================
# Row 2: H0 (Non-causal: neighbors -> random)
# ==========================================
ax_map2 = axes[1, 0]
ax_auc2 = axes[1, 1]

# Draw a divider and background shades to clearly separate the two state spaces
ax_map2.axvspan(-1, 4.5, facecolor='#f4f7f9', alpha=0.6, zorder=0)
ax_map2.axvspan(4.5, 10, facecolor='#f4f9f4', alpha=0.6, zorder=0)  # Subtle green background for MX
ax_map2.axvline(x=4.5, color='gray', linestyle='-.', lw=2, alpha=0.5, zorder=1)

# MY space (Left)
circle_y2 = Circle((2, 5), 1.5, edgecolor=c_y, facecolor='#e6f2ff', lw=2.5)
ax_map2.add_patch(circle_y2)
ax_map2.scatter(2, 5, color=c_y, s=pt_center, zorder=6)
ax_map2.text(2, 5.3, '$Y(t)$', ha='center', va='bottom', fontsize=text_fs, fontweight='bold', color=c_y, bbox=text_bbox, zorder=7)
ax_map2.scatter(all_y_x, all_y_y, color=c_y, s=pt_nbr, zorder=5)
ax_map2.text(2, 3.1, '$M_Y$', ha='center', va='top', fontsize=manifold_fs, fontweight='bold', color=c_y)

# MX space (Right)
circle_x2_r2 = Circle((7, 5), r2, edgecolor=c_x2, facecolor='#f0fff0', lw=3, linestyle=':', alpha=0.7)
circle_x2_r1 = Circle((7, 5), r1, edgecolor=c_x2, facecolor='#e6ffe6', lw=2.5, linestyle='--', alpha=0.9)
ax_map2.add_patch(circle_x2_r2)
ax_map2.add_patch(circle_x2_r1)
ax_map2.text(7, 2.5, '$M_X$', ha='center', va='top', fontsize=manifold_fs, fontweight='bold', color=c_x2)

# Mapped points (Original 6 points are randomized)
# Adjusted rand_radii to ensure no points cross the x=4.5 dividing line
rand_radii = np.array([0.8, 1.8, 2.2, 2.3, 2.1, 1.4]) 
rand_angles = np.array([0.2, 1.1, 2.3, 3.5, 4.8, 5.7])
x_rand_orig_x = 7 + rand_radii * np.cos(rand_angles)
x_rand_orig_y = 5 + rand_radii * np.sin(rand_angles)

# 2 additional boundary points in X space for H0 (placed at different random angles to emphasize non-causality)
angle_pt1_h0 = -np.pi / 5
angle_pt2_h0 = 4 * np.pi / 5
x_add_x1_h0 = 7 + r1 * np.cos(angle_pt1_h0)
x_add_y1_h0 = 5 + r1 * np.sin(angle_pt1_h0)
x_add_x2_h0 = 7 + r2 * np.cos(angle_pt2_h0)
x_add_y2_h0 = 5 + r2 * np.sin(angle_pt2_h0)

all_x2_x = np.concatenate([x_rand_orig_x, [x_add_x1_h0, x_add_x2_h0]])
all_x2_y = np.concatenate([x_rand_orig_y, [x_add_y1_h0, x_add_y2_h0]])

ax_map2.scatter(all_x2_x, all_x2_y, color=c_x2, s=pt_nbr, zorder=5, marker='s')

ax_map2.scatter(7, 5, color=c_x2, s=pt_center, zorder=6, marker='s')
ax_map2.text(7, 5.3, '$X(t)$', ha='center', va='bottom', fontsize=text_fs, fontweight='bold', color=c_x2, bbox=text_bbox, zorder=7)

# Radii arrows
ax_map2.annotate('', xy=(7 + r1 * np.cos(angle_r1), 5 + r1 * np.sin(angle_r1)), 
                 xytext=(7, 5), arrowprops=dict(arrowstyle='-|>', color=c_x2, lw=2), zorder=4)
ax_map2.text(7 + (r1 / 2) * np.cos(angle_r1) + 0.15, 5 + (r1 / 2) * np.sin(angle_r1) - 0.1, '$r_1$', 
             color=c_x2, fontweight='bold', fontsize=text_fs, ha='center', va='center', bbox=text_bbox, zorder=7)

ax_map2.annotate('', xy=(7 + r2 * np.cos(angle_r2), 5 + r2 * np.sin(angle_r2)), 
                 xytext=(7, 5), arrowprops=dict(arrowstyle='-|>', color=c_x2, lw=2), zorder=4)
ax_map2.text(7 + (r1 + (r2 - r1) / 2) * np.cos(angle_r2), 5 + (r1 + (r2 - r1) / 2) * np.sin(angle_r2) - 0.15, '$r_2$', 
             color=c_x2, fontweight='bold', fontsize=text_fs, ha='center', va='center', bbox=text_bbox, zorder=7)

# Draw arrows for all 8 points
for i in range(len(all_y_x)):
    arrow = FancyArrowPatch((all_y_x[i], all_y_y[i]), (all_x2_x[i], all_x2_y[i]),
                            arrowstyle='->', mutation_scale=mut_scale, color=c_line, lw=arrow_lw, alpha=0.8, zorder=3)
    ax_map2.add_patch(arrow)

ax_map2.set_xlim(0, 9.5)
ax_map2.set_ylim(2.2, 7.5)
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
ax_auc2.set_xlabel('kNN Rank Fraction ($r$)', fontsize=label_fs, fontweight='bold')
ax_auc2.set_ylabel('True Positive Rate', fontsize=label_fs, fontweight='bold')
ax_auc2.set_title(r'$AUC(H_0) \approx 0.5$', fontsize=ax_title_fs, fontweight='bold', color=c_x2)
ax_auc2.grid(True, linestyle=':', alpha=0.6)

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    plt.tight_layout()
plt.savefig('cmc_schematic_ppt_separated.png', dpi=300)