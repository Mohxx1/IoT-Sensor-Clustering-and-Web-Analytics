import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

def analyze_web_traffic():
    print("--- Running Web Traffic Analytics ---")
    # Generate synthetic website user activity data
    hours = np.arange(0, 24)
    page_views = np.random.poisson(lam=150, size=24) + (np.sin(hours / 24 * 2 * np.pi) * 80).astype(int)
    bounce_rate = np.clip(np.random.normal(loc=45, scale=10, size=24), 20, 85)

    df_web = pd.DataFrame({
        'Hour': hours,
        'Page_Views': np.abs(page_views),
        'Bounce_Rate (%)': np.round(bounce_rate, 2)
    })

    print(df_web.describe())

    # Plot Web Traffic
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Hour of the Day')
    ax1.set_ylabel('Page Views', color='tab:blue')
    ax1.plot(df_web['Hour'], df_web['Page_Views'], color='tab:blue', marker='o', label='Page Views')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(hours)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Bounce Rate (%)', color='tab:red')
    ax2.plot(df_web['Hour'], df_web['Bounce_Rate (%)'], color='tab:red', linestyle='--', marker='x', label='Bounce Rate')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('24-Hour Website User Activity & Bounce Rate Analysis')
    fig.tight_layout()
    plt.savefig('web_traffic_analysis.png')
    plt.close()
    print("Saved 'web_traffic_analysis.png'\n")

def cluster_iot_sensors():
    print("--- Running IoT Sensor K-Means Clustering ---")
    # Generate synthetic IoT environmental sensor readings (Temperature & Humidity)
    # Cluster 1: Normal Operating Zone, Cluster 2: High Temp Alert, Cluster 3: High Humidity Zone
    c1 = np.random.normal(loc=[22, 45], scale=[2, 5], size=(100, 2))
    c2 = np.random.normal(loc=[35, 30], scale=[2.5, 4], size=(100, 2))
    c3 = np.random.normal(loc=[25, 80], scale=[2, 5], size=(100, 2))

    sensor_data = np.vstack([c1, c2, c3])
    df_iot = pd.DataFrame(sensor_data, columns=['Temperature_C', 'Humidity_Pct'])

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_iot)

    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_iot['Cluster'] = kmeans.fit_predict(scaled_features)

    print("Cluster distribution:")
    print(df_iot['Cluster'].value_counts())

    # Plot Clusters
    plt.figure(figsize=(8, 6))
    colors = ['#2b5c8f', '#d95f02', '#2ca02c']
    labels = ['Normal Operation', 'High Temp Alert', 'High Humidity Zone']

    for i in range(3):
        subset = df_iot[df_iot['Cluster'] == i]
        plt.scatter(subset['Temperature_C'], subset['Humidity_Pct'], s=50, c=colors[i], label=labels[i], alpha=0.8)

    # Plot Centroids (inverse transformed)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')

    plt.title('K-Means Clustering of IoT Sensor Telemetry')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Relative Humidity (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('iot_kmeans_clusters.png')
    plt.close()
    print("Saved 'iot_kmeans_clusters.png'")

if __name__ == "__main__":
    analyze_web_traffic()
    cluster_iot_sensors()