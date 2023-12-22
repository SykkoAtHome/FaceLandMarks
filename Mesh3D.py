import matplotlib.pyplot as plt


def landmarks_to_3d_object(frame_landmarks_df):
    x = frame_landmarks_df['x'].tolist()
    y = frame_landmarks_df['y'].tolist()
    z = frame_landmarks_df['z'].tolist()

    return x, y, z


def create_3d_point_cloud(video_file):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = landmarks_to_3d_object(video_file.landmarks_to_dataframe())
    ax.scatter(x, y, z, c='r', marker='o')

    plt.show()


def create_3d_geometry(video_file):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = video_file.dataframe['x'].tolist()
    y = video_file.dataframe['y'].tolist()
    z = video_file.dataframe['z'].tolist()

    ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='k')

    plt.show()
