import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D

data_x = [0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.8, 0.9, 1.0]
data_y = [2.0, 1.6, 1.8, 1.2, 1.4, 1.0, 1.2, 1.0, 0.8, 0.8, 0.6]

# Korrelation berechnen
corr_coef, p_value = stats.pearsonr(data_x, data_y)

# Schicker Plot im distancePlot-Stil
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot mit ähnlichem Stil wie distancePlot
ax.scatter(data_x, data_y, s=100, alpha=0.7, color='steelblue', 
           edgecolors='darkblue', linewidth=1.5, zorder=3, label='Datenpunkte')

# Trendlinie hinzufügen
z = np.polyfit(data_x, data_y, 1)
p = np.poly1d(z)
# x_trend = np.linspace(min(data_x), max(data_x), 100)
# ax.plot(x_trend, p(x_trend), '--', color='red', linewidth=2, alpha=0.8, 
#         label=f'Trendlinie (r={corr_coef:.3f}, p={p_value:.3f})')

# Titel und Achsenbeschriftung im distancePlot-Stil
ax.set_title('Zusammenhang zwischen Global Epsilon und Global Rho Scale\nfür die besten Ergebnisse', 
             fontsize=12, pad=15)
ax.set_xlabel('Global Epsilon', fontsize=11)
ax.set_ylabel('Global Rho Scale', fontsize=11)

# Grid im distancePlot-Stil (nur horizontale Linien)
ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.7)

# Achsen-Bereich optimieren
ax.set_xlim(0.25, 1.05)
ax.set_ylim(0.5, 2.1)

# # Statistik-Text hinzufügen (ähnlich wie bei distancePlot)
# stats_text = f'n = {len(data_x)}\nr = {corr_coef:.3f}\np = {p_value:.3f}'
# ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
#         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Layout optimieren und speichern
plt.tight_layout()
plt.savefig('correlation_plot.png', dpi=300, bbox_inches='tight')