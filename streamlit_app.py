import streamlit as st
import pandas as pd
import math
import numpy as np
from sklearn.neighbors import KernelDensity
from pathlib import Path
import matplotlib.pyplot as plt

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Anomaly Detection with KDE',
    page_icon=':mag:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :mag: Anomaly Detection with KDE
'''

def random_values(n, min, max):
    return np.random.rand(n)*(max-min) + min

np.random.seed(seed=0)

# Add some spacing
''
''
min_value = -30
max_value = 30
min_value, max_value = st.slider(
    'Select min and max value',
    min_value=-30,
    max_value=30,
    value=[min_value, max_value])

# st.write(f"{min_value}, {max_value}")
threadhold_percent = 10
bandwidth = 1

n_cluster = 1
n_cluster = st.number_input("Select number of cluster", value=n_cluster, step=1, min_value=1)

cols = st.columns(n_cluster, gap="medium")
pts_min = [min_value for i in range(n_cluster)]
pts_max = [max_value for i in range(n_cluster)]
n_pts = [1 for i in range(n_cluster)]
clusters = [[] for i in range(n_cluster)]


for i in range(n_cluster):
    with cols[i]:
        pts_min[i],pts_max[i] = st.slider(f"Min and max value for {i}th cluster", 
                                    min_value=min_value, max_value=max_value,
                                    value=[pts_min[i],pts_max[i]])
        n_pts[i] = st.number_input("number of point", value=n_pts[i], key=f"{i}_3")
        clusters[i] = random_values(n_pts[i], pts_min[i], pts_max[i])

pts = np.concatenate(clusters)
anomaly_pts = np.random.rand(int(pts.size*threadhold_percent/100))*(max_value-min_value) + min_value
pts = np.concatenate([pts, anomaly_pts])



''
''


bandwidth = st.slider(
    'Select bandwidth',
    min_value=1,
    max_value=100,
    value=bandwidth)

threadhold_percent = st.slider(
    'Select anomaly threadhold (percent)',
    min_value=1,
    max_value=100,
    value=threadhold_percent)

kernel = st.selectbox(
    "Select kernel",
    ('gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'),
)

kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)  # Chọn kernel Gaussian và băng thông là 0.5
kde.fit(pts.reshape(-1, 1))  # Reshape data thành dạng 2D cho scikit-learn
log_dens = kde.score_samples(pts.reshape(-1, 1))
threshold = np.percentile(log_dens, threadhold_percent)
outliers = pts[log_dens < threshold]

# Vẽ mật độ KDE
fig, ax = plt.subplots()

x_grid = np.linspace(min_value, max_value, 1000)
log_dens_grid = kde.score_samples(x_grid.reshape(-1, 1))
ax.plot(x_grid, np.exp(log_dens_grid), label='KDE Density Estimation', color='blue')


# Vẽ các điểm dữ liệu bình thường
ax.scatter(pts, np.zeros_like(pts), c='green', label='Normal Data', alpha=0.5)

# Vẽ các điểm bất thường
ax.scatter(outliers, np.zeros_like(outliers), c='red', label='Anomalies', alpha=1)

# Thêm nhãn và tiêu đề
ax.legend()

st.pyplot(fig)
