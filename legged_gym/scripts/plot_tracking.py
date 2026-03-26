import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def main():
    file_path = "circle_tracking_data.npy"
    if not os.path.exists(file_path):
        # Try checking in the current script directory if not in PWD
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "circle_tracking_data.npy")
        
    print(f"Loading data from {file_path}...")
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print("Data file 'circle_tracking_data.npy' not found. Please run 'play_b2z1posforce_circle.py' first and let it run for a few seconds.")
        return

    print(f"Loaded data with shape: {data.shape}")
    
    # data structure: [target_x, target_y, target_z, actual_x, actual_y, actual_z]
    target_pos = data[:, 0:3]
    actual_pos = data[:, 3:6]
    
    time_steps = np.arange(len(data))
    
    # Figure 1: 3D Trajectory
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], label='Target (Body-Rel)', color='red', linestyle='--')
    ax1.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2], label='Actual (Body-Rel)', color='blue')
    
    # Start/End points
    ax1.scatter(target_pos[0,0], target_pos[0,1], target_pos[0,2], c='green', marker='o', s=50, label='Start')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D EE Trajectory (Body-Relative Frame)')
    ax1.legend()
    
    # Equal aspect ratio for 3D plot is tricky in matplotlib, but we can try to set limits
    x_limits = [min(np.min(target_pos[:,0]), np.min(actual_pos[:,0])), max(np.max(target_pos[:,0]), np.max(actual_pos[:,0]))]
    y_limits = [min(np.min(target_pos[:,1]), np.min(actual_pos[:,1])), max(np.max(target_pos[:,1]), np.max(actual_pos[:,1]))]
    z_limits = [min(np.min(target_pos[:,2]), np.min(actual_pos[:,2])), max(np.max(target_pos[:,2]), np.max(actual_pos[:,2]))]
    
    ax1.set_xlim(x_limits)
    ax1.set_ylim(y_limits)
    ax1.set_zlim(z_limits)

    # Figure 2: XYZ vs Time
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(time_steps, target_pos[:, 0], 'r--', label='Target X', alpha=0.6)
    ax2.plot(time_steps, actual_pos[:, 0], 'r-', label='Actual X')
    
    ax2.plot(time_steps, target_pos[:, 1], 'g--', label='Target Y', alpha=0.6)
    ax2.plot(time_steps, actual_pos[:, 1], 'g-', label='Actual Y')
    
    ax2.plot(time_steps, target_pos[:, 2], 'b--', label='Target Z', alpha=0.6)
    ax2.plot(time_steps, actual_pos[:, 2], 'b-', label='Actual Z')
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Component Tracking')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('trajectory_plot.png')
    print("Plot saved to 'trajectory_plot.png'")
    # plt.show() # Uncomment if running in a windowed environment

if __name__ == "__main__":
    main()
