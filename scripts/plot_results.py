import matplotlib.pyplot as plt
import numpy as np
import os

# 创建保存目录
os.makedirs('docs/figures', exist_ok=True)

# 数据
shots = ['5-shot', '10-shot']
supervised = [17.59, 22.63]
linear = [59.88, 68.09]
finetune = [49.25, 63.79]

x = np.arange(len(shots))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x - width, supervised, width, label='Supervised', color='#d62728')
bars2 = ax.bar(x, linear, width, label='Linear', color='#2ca02c')
bars3 = ax.bar(x + width, finetune, width, label='Full Finetune', color='#1f77b4')

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Few-shot Classification on CIFAR-10', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(shots, fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, 80)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('docs/figures/comparison.png', dpi=150)
print("图表已保存至 docs/figures/comparison.png")