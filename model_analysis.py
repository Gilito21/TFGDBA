# Function to load meshes

import numpy as np
import trimesh
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
import os

def load_meshes(original_path='porsche_original.obj', damaged_path='porsche_damaged.obj'):
    try:
        # Check if files exist
        if os.path.exists(original_path) and os.path.exists(damaged_path):
            print(f"Loading meshes from {original_path} and {damaged_path}")
            mesh_original = trimesh.load(original_path)
            mesh_damaged = trimesh.load(damaged_path)

            # Handle scene objects
            if isinstance(mesh_original, trimesh.Scene):
                mesh_original = trimesh.util.concatenate(mesh_original.geometry.values())
            if isinstance(mesh_damaged, trimesh.Scene):
                mesh_damaged = trimesh.util.concatenate(mesh_damaged.geometry.values())

            print(f"Original mesh: {len(mesh_original.vertices)} vertices, {len(mesh_original.faces)} faces")
            print(f"Damaged mesh: {len(mesh_damaged.vertices)} vertices, {len(mesh_damaged.faces)} faces")

            return mesh_original, mesh_damaged
        else:
            print("Files not found.")
            raise FileNotFoundError("Mesh files not found")
    except Exception as e:
        print(f"Error loading meshes: {e}")
        return None, None

# Analysis functions (same as before)
def analyze_damage(mesh_original, mesh_damaged):
    if len(mesh_original.vertices) != len(mesh_damaged.vertices):
        print("Meshes have different vertex counts. Cannot compute detailed analysis.")
        return None, None

    # Calculate distances between corresponding vertices
    distances = np.linalg.norm(mesh_original.vertices - mesh_damaged.vertices, axis=1)

    # Get statistical measures to determine significant damage
    mean_diff = np.mean(distances)
    std_diff = np.std(distances)
    max_diff = np.max(distances)

    # Set multiple thresholds for damage severity classification
    severe_threshold = np.percentile(distances, 99)  # Top 1% = severe damage
    moderate_threshold = np.percentile(distances, 97)  # Top 3% = moderate damage
    mild_threshold = np.percentile(distances, 95)     # Top 5% = mild damage

    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Std deviation: {std_diff:.6f}")
    print(f"Maximum difference: {max_diff:.6f}")
    print(f"Severe damage threshold (99th percentile): {severe_threshold:.6f}")
    print(f"Moderate damage threshold (97th percentile): {moderate_threshold:.6f}")
    print(f"Mild damage threshold (95th percentile): {mild_threshold:.6f}")

    # Find indices of damaged vertices at different severity levels
    severe_indices = np.where(distances > severe_threshold)[0]
    moderate_indices = np.where((distances > moderate_threshold) & (distances <= severe_threshold))[0]
    mild_indices = np.where((distances > mild_threshold) & (distances <= moderate_threshold))[0]

    print(f"Vertices with severe damage: {len(severe_indices)}")
    print(f"Vertices with moderate damage: {len(moderate_indices)}")
    print(f"Vertices with mild damage: {len(mild_indices)}")

    # Get all damage points above the mild threshold for clustering
    damage_indices = np.where(distances > mild_threshold)[0]

    if len(damage_indices) == 0:
        print("No significant damage detected.")
        return distances, None

    damage_points = mesh_damaged.vertices[damage_indices]
    damage_values = distances[damage_indices]

    # Use DBSCAN for clustering damage areas
    if len(damage_points) < 5:
        print("Not enough damage points detected for clustering")
        return distances, None

    # Determine eps based on the mesh scale
    bbox = mesh_damaged.bounds
    diagonal = np.linalg.norm(bbox[1] - bbox[0])
    eps = diagonal * 0.05  # 5% of diagonal for clustering

    # Apply clustering
    clustering = DBSCAN(eps=eps, min_samples=5).fit(damage_points)
    labels = clustering.labels_

    # Count number of clusters (excluding noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Detected {n_clusters} damage clusters")

    # Collect cluster information
    damage_clusters = []
    for label in set(labels):
        if label == -1:  # Skip noise points
            continue

        # Get points in this cluster
        mask = labels == label
        cluster_points = damage_points[mask]
        cluster_values = damage_values[mask]
        cluster_indices = damage_indices[mask]

        # Calculate cluster statistics
        center = np.mean(cluster_points, axis=0)
        size = len(cluster_points)
        max_damage = np.max(cluster_values)
        avg_damage = np.mean(cluster_values)
        total_damage = np.sum(cluster_values)

        # Determine severity based on maximum damage in the cluster
        if max_damage > severe_threshold:
            severity = "Severe"
        elif max_damage > moderate_threshold:
            severity = "Moderate"
        else:
            severity = "Mild"

        # Calculate volume of the damage (approximation)
        try:
            # Create a convex hull from the damage points
            hull = trimesh.convex.convex_hull(cluster_points)
            volume = hull.volume
            surface_area = hull.area
            damage_mesh = hull  # Store the convex hull as the damage mesh
        except Exception as e:
            print(f"Could not calculate convex hull for cluster {len(damage_clusters) + 1}: {e}")
            volume = 0
            surface_area = 0
            damage_mesh = None

            # Alternative: try alpha shape if available
            try:
                # If we can't make a convex hull, create a simple point cloud trimesh
                damage_mesh = trimesh.points.PointCloud(cluster_points)
                volume = 0
                surface_area = 0
            except:
                damage_mesh = None

        damage_clusters.append({
            'id': len(damage_clusters) + 1,
            'center': center,
            'size': size,
            'max_damage': max_damage,
            'avg_damage': avg_damage,
            'total_damage': total_damage,
            'volume': volume,
            'surface_area': surface_area,
            'severity': severity,
            'points': cluster_points,
            'values': cluster_values,
            'indices': cluster_indices,
            'mesh': damage_mesh,
            'label': f"Damage Area {len(damage_clusters) + 1} ({severity})"
        })

    # Sort clusters by severity and then by size
    damage_clusters.sort(key=lambda x: (0 if x['severity'] == 'Severe' else
                                       (1 if x['severity'] == 'Moderate' else 2),
                                        -x['size']))

    # Renumber clusters after sorting
    for i, cluster in enumerate(damage_clusters):
        cluster['id'] = i + 1
        cluster['label'] = f"Damage Area {i + 1} ({cluster['severity']})"

    return distances, damage_clusters

# Create a submesh from vertices of a specific cluster
def create_damage_submesh(mesh, cluster_indices):
    """Create a submesh containing only the faces that include the damaged vertices"""
    damaged_vertices = set(cluster_indices)

    # Find faces that contain at least one damaged vertex
    damaged_faces = []
    for i, face in enumerate(mesh.faces):
        if face[0] in damaged_vertices or face[1] in damaged_vertices or face[2] in damaged_vertices:
            damaged_faces.append(i)

    # Create a submesh with only the damaged faces
    if damaged_faces:
        submesh = mesh.submesh([damaged_faces], append=True)
        return submesh
    else:
        return None

# Enhanced visualization with improved layout
def create_improved_visualization(mesh_original, mesh_damaged, distances, damage_clusters):
    # Create figure with a larger size
    fig = go.Figure()

    # Add the original mesh (initially invisible)
    fig.add_trace(go.Mesh3d(
        x=mesh_original.vertices[:, 0],
        y=mesh_original.vertices[:, 1],
        z=mesh_original.vertices[:, 2],
        i=mesh_original.faces[:, 0],
        j=mesh_original.faces[:, 1],
        k=mesh_original.faces[:, 2],
        color='blue',
        opacity=0.5,
        name='Original Mesh',
        visible=False
    ))

    # Add the damaged mesh with color based on damage intensity
    fig.add_trace(go.Mesh3d(
        x=mesh_damaged.vertices[:, 0],
        y=mesh_damaged.vertices[:, 1],
        z=mesh_damaged.vertices[:, 2],
        i=mesh_damaged.faces[:, 0],
        j=mesh_damaged.faces[:, 1],
        k=mesh_damaged.faces[:, 2],
        intensity=distances,
        colorscale='Viridis',
        opacity=0.8,
        name='Damaged Mesh',
        showscale=True,
        colorbar=dict(
            title='Deformation',
            titleside='right',
            thickness=20,
            len=0.6,
            y=0.5
        ),
        visible=True
    ))

    # Add damage meshes as separate traces
    if damage_clusters:
        for damage in damage_clusters:
            # Determine color based on severity
            if damage['severity'] == 'Severe':
                color = 'red'
            elif damage['severity'] == 'Moderate':
                color = 'orange'
            else:  # Mild
                color = 'yellow'

            # Create submesh for this damage area
            try:
                damage_submesh = create_damage_submesh(mesh_damaged, damage['indices'])

                if damage_submesh is not None:
                    # Add damage submesh
                    fig.add_trace(go.Mesh3d(
                        x=damage_submesh.vertices[:, 0],
                        y=damage_submesh.vertices[:, 1],
                        z=damage_submesh.vertices[:, 2],
                        i=damage_submesh.faces[:, 0],
                        j=damage_submesh.faces[:, 1],
                        k=damage_submesh.faces[:, 2],
                        color=color,
                        opacity=0.9,
                        name=f"Damage {damage['id']} Mesh",
                        hoverinfo='name',
                        visible=False  # Initially hidden
                    ))
                else:
                    # If we couldn't create a proper submesh, fallback to the convex hull
                    if damage['mesh'] is not None and hasattr(damage['mesh'], 'vertices') and hasattr(damage['mesh'], 'faces'):
                        hull_mesh = damage['mesh']
                        fig.add_trace(go.Mesh3d(
                            x=hull_mesh.vertices[:, 0],
                            y=hull_mesh.vertices[:, 1],
                            z=hull_mesh.vertices[:, 2],
                            i=hull_mesh.faces[:, 0],
                            j=hull_mesh.faces[:, 1],
                            k=hull_mesh.faces[:, 2],
                            color=color,
                            opacity=0.9,
                            name=f"Damage {damage['id']} Hull",
                            hoverinfo='name',
                            visible=False  # Initially hidden
                        ))
            except Exception as e:
                print(f"Error creating damage mesh for cluster {damage['id']}: {e}")

            # Add text label at damage center
            fig.add_trace(go.Scatter3d(
                x=[damage['center'][0]],
                y=[damage['center'][1]],
                z=[damage['center'][2]],
                mode='markers+text',
                marker=dict(size=10, color=color, symbol='circle'),
                text=[f"Area {damage['id']}"],  # Shorter text for less clutter
                textposition="top center",
                name=damage['label'],
                showlegend=True,
                hoverinfo='text',
                hovertext=[f"<b>Damage Area {damage['id']}</b><br>" +
                           f"Severity: {damage['severity']}<br>" +
                           f"Max deformation: {damage['max_damage']:.4f}<br>" +
                           f"Avg deformation: {damage['avg_damage']:.4f}<br>" +
                           f"Affected points: {damage['size']}<br>" +
                           f"Approx. volume: {damage['volume']:.4f}<br>" +
                           f"Approx. surface area: {damage['surface_area']:.4f}"],
                visible=True  # Labels are always visible
            ))

    # Count how many traces we have (needed for visibility settings)
    total_traces = len(fig.data)
    mesh_original_idx = 0
    mesh_damaged_idx = 1
    damage_mesh_indices = list(range(2, total_traces))
    damage_label_indices = []
    damage_submesh_indices = []

    # Separate damage labels from damage meshes
    for i in damage_mesh_indices:
        if 'mode' in fig.data[i] and fig.data[i]['mode'] == 'markers+text':
            damage_label_indices.append(i)
        else:
            damage_submesh_indices.append(i)

    # Create visibility settings for each view
    damaged_only_vis = [False, True] + [True if i in damage_label_indices else False for i in range(2, total_traces)]
    original_only_vis = [True, False] + [True if i in damage_label_indices else False for i in range(2, total_traces)]
    both_meshes_vis = [True, True] + [True if i in damage_label_indices else False for i in range(2, total_traces)]
    damage_areas_only_vis = [False, False] + [True for i in range(2, total_traces)]
    original_and_damage_vis = [True, False] + [True for i in range(2, total_traces)]
    damaged_and_damage_vis = [False, True] + [True for i in range(2, total_traces)]

    # Add buttons for toggling views
    buttons = [
        {
            'label': "Damaged Mesh Only",
            'method': "update",
            'args': [{"visible": damaged_only_vis},
                     {"title": "Damage Analysis - Damaged Mesh Only"}]
        },
        {
            'label': "Original Mesh Only",
            'method': "update",
            'args': [{"visible": original_only_vis},
                     {"title": "Damage Analysis - Original Mesh Only"}]
        },
        {
            'label': "Both Meshes",
            'method': "update",
            'args': [{"visible": both_meshes_vis},
                     {"title": "Damage Analysis - Both Meshes Comparison"}]
        },
        {
            'label': "Damage Areas Only",
            'method': "update",
            'args': [{"visible": damage_areas_only_vis},
                     {"title": "Damage Analysis - Damage Areas Only"}]
        },
        {
            'label': "Original + Damage Areas",
            'method': "update",
            'args': [{"visible": original_and_damage_vis},
                     {"title": "Damage Analysis - Original Mesh with Damage Areas"}]
        },
        {
            'label': "Damaged + Damage Areas",
            'method': "update",
            'args': [{"visible": damaged_and_damage_vis},
                     {"title": "Damage Analysis - Damaged Mesh with Damage Areas"}]
        }
    ]

    # Update layout with improved positioning
    fig.update_layout(
        # Move buttons to the top-right corner
        updatemenus=[
            {
                'type': 'buttons',
                'direction': 'down',  # Changed to dropdown
                'x': 1.05,  # Move to right edge
                'y': 1.0,   # Top of the plot
                'xanchor': 'right',
                'yanchor': 'top',
                'buttons': buttons,
                'showactive': True,
                'bgcolor': 'rgba(255, 255, 255, 0.9)',  # Semi-transparent background
                'bordercolor': 'rgba(0, 0, 0, 0.5)',
                'font': {'size': 10}  # Smaller font
            }
        ],
        title={
            'text': 'Damage Analysis',
            'y': 0.95,  # Move title up slightly
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)  # Adjust default camera position
            ),
            # Make the 3D scene larger
            domain=dict(x=[0, 1], y=[0, 1])
        ),
        # Adjust margins to maximize visualization space
        margin=dict(l=10, r=100, t=60, b=10),
        # Make the figure larger
        height=700,  # Taller figure
        width=1000,  # Wider figure
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.7)'  # Semi-transparent background
        ),
        # Add some annotation instructions
        annotations=[
            dict(
                x=0.98,
                y=0.01,
                xref="paper",
                yref="paper",
                text="Use mouse to rotate, scroll to zoom",
                showarrow=False,
                font=dict(size=10, color="grey")
            )
        ]
    )

    return fig

# Create a summary table visualization of damage statistics
def create_damage_summary(damage_clusters):
    if not damage_clusters:
        return None

    # Create a table of damage statistics
    table_data = [
        ['<b>Damage Area</b>', '<b>Severity</b>', '<b>Max Deformation</b>', '<b>Avg Deformation</b>',
         '<b>Affected Points</b>', '<b>Approx. Volume</b>', '<b>Approx. Surface Area</b>']
    ]

    for damage in damage_clusters:
        table_data.append([
            f"Area {damage['id']}",
            damage['severity'],
            f"{damage['max_damage']:.4f}",
            f"{damage['avg_damage']:.4f}",
            f"{damage['size']}",
            f"{damage['volume']:.4f}",
            f"{damage['surface_area']:.4f}"
        ])

    # Total row
    total_points = sum(d['size'] for d in damage_clusters)
    avg_max_damage = np.mean([d['max_damage'] for d in damage_clusters])
    avg_avg_damage = np.mean([d['avg_damage'] for d in damage_clusters])
    total_volume = sum(d['volume'] for d in damage_clusters)
    total_surface = sum(d['surface_area'] for d in damage_clusters)

    table_data.append([
        '<b>TOTAL</b>',
        '',
        f"<b>{avg_max_damage:.4f}</b>",
        f"<b>{avg_avg_damage:.4f}</b>",
        f"<b>{total_points}</b>",
        f"<b>{total_volume:.4f}</b>",
        f"<b>{total_surface:.4f}</b>"
    ])

    # Create the table visualization
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Damage Area', 'Severity', 'Max Deformation', 'Avg Deformation',
                    'Affected Points', 'Approx. Volume', 'Approx. Surface Area'],
            font=dict(size=12, color='white'),
            fill_color='darkblue',
            align='center'
        ),
        cells=dict(
            values=list(zip(*table_data))[1:],  # Transpose data
            font=dict(size=11),
            fill_color=[['lightgray' if i % 2 == 0 else 'white' for i in range(len(table_data))]] * 6,
            align='center'
        ),
        columnwidth=[1, 1, 1.5, 1.5, 1.5, 1.5, 1.5]
    )])

    fig.update_layout(
        title='Damage Analysis Summary',
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
        height=150 + 30 * len(damage_clusters)
    )

    return fig

# Main function to detect and visualize damage
def detect_and_visualize_damage():
    # 1. Load the meshes
    mesh_original, mesh_damaged = load_meshes()
    if mesh_original is None or mesh_damaged is None:
        print("Failed to load meshes")
        return

    # 2. Analyze damage
    distances, damage_clusters = analyze_damage(mesh_original, mesh_damaged)

    # 3. Generate damage statistics
    if damage_clusters:
        # Calculate overall damage statistics
        total_affected_points = sum(cluster['size'] for cluster in damage_clusters)
        pct_affected = (total_affected_points / len(mesh_damaged.vertices)) * 100

        # Print damage summary
        print("\n===== DAMAGE ANALYSIS SUMMARY =====")
        print(f"Total damage clusters detected: {len(damage_clusters)}")
        print(f"Total affected vertices: {total_affected_points} ({pct_affected:.2f}% of mesh)")

        clusters_by_severity = {
            'Severe': len([c for c in damage_clusters if c['severity'] == 'Severe']),
            'Moderate': len([c for c in damage_clusters if c['severity'] == 'Moderate']),
            'Mild': len([c for c in damage_clusters if c['severity'] == 'Mild'])
        }

        print("\nDamage clusters by severity:")
        for severity, count in clusters_by_severity.items():
            print(f"  {severity}: {count}")

        print("\nDetailed damage cluster information:")
        for i, damage in enumerate(damage_clusters):
            print(f"\nDamage Area {damage['id']} ({damage['severity']}):")
            print(f"  Max deformation: {damage['max_damage']:.4f}")
            print(f"  Average deformation: {damage['avg_damage']:.4f}")
            print(f"  Affected points: {damage['size']} vertices")
            print(f"  Approximate volume: {damage['volume']:.4f}")
            print(f"  Approximate surface area: {damage['surface_area']:.4f}")
    else:
        print("No significant damage detected")

    # 4. Create enhanced interactive visualization
    fig_3d = create_improved_visualization(mesh_original, mesh_damaged, distances, damage_clusters)
    fig_3d.show()

    # 5. Create damage summary table
    if damage_clusters:
        fig_table = create_damage_summary(damage_clusters)
        fig_table.show()

    return damage_clusters
