"""
Geospatial visualization utilities for India map.

Creates interactive maps showing enrollment data by state and district.
"""

import pandas as pd
import numpy as np
import folium
from folium import plugins
import json

# India state coordinates (approximate centroids)
INDIA_STATE_COORDINATES = {
    'Andhra Pradesh': [15.9129, 79.7400],
    'Arunachal Pradesh': [28.2180, 94.7278],
    'Assam': [26.2006, 92.9376],
    'Bihar': [25.0961, 85.3131],
    'Chhattisgarh': [21.2787, 81.8661],
    'Goa': [15.2993, 74.1240],
    'Gujarat': [22.2587, 71.1924],
    'Haryana': [29.0588, 76.0856],
    'Himachal Pradesh': [31.1048, 77.1734],
    'Jharkhand': [23.6102, 85.2799],
    'Karnataka': [15.3173, 75.7139],
    'Kerala': [10.8505, 76.2711],
    'Madhya Pradesh': [22.9734, 78.6569],
    'Maharashtra': [19.7515, 75.7139],
    'Manipur': [24.6637, 93.9063],
    'Meghalaya': [25.4670, 91.3662],
    'Mizoram': [23.1645, 92.9376],
    'Nagaland': [26.1584, 94.5624],
    'Odisha': [20.9517, 85.0985],
    'Punjab': [31.1471, 75.3412],
    'Rajasthan': [27.0238, 74.2179],
    'Sikkim': [27.5330, 88.5122],
    'Tamil Nadu': [11.1271, 78.6569],
    'Telangana': [18.1124, 79.0193],
    'Tripura': [23.9408, 91.9882],
    'Uttar Pradesh': [26.8467, 80.9462],
    'Uttarakhand': [30.0668, 79.0193],
    'West Bengal': [22.9868, 87.8550],
    'Andaman And Nicobar Islands': [11.7401, 92.6586],
    'Chandigarh': [30.7333, 76.7794],
    'Dadra And Nagar Haveli And Daman And Diu': [20.1809, 73.0169],
    'Delhi': [28.7041, 77.1025],
    'Jammu And Kashmir': [33.7782, 76.5762],
    'Ladakh': [34.1526, 77.5771],
    'Lakshadweep': [10.5667, 72.6417],
    'Puducherry': [11.9416, 79.8083]
}


def create_india_enrollment_map(df, metric='total_enrolment', zoom_start=5):
    """
    Create an interactive India map showing enrollment data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with state-level aggregated data
    metric : str
        Metric to visualize ('total_enrolment', 'total_updates', 'update_pressure_index')
    zoom_start : int
        Initial zoom level
        
    Returns:
    --------
    folium.Map
        Interactive map object
    """
    # Aggregate data by state
    state_data = df.groupby('state').agg({
        'total_enrolment': 'sum',
        'total_updates': 'sum',
        'update_pressure_index': 'mean',
        'total_demo_updates': 'sum',
        'total_bio_updates': 'sum'
    }).reset_index()
    
    # Create base map centered on India
    india_map = folium.Map(
        location=[20.5937, 78.9629],  # Center of India
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )
    
    # Add markers for each state
    for _, row in state_data.iterrows():
        state_name = row['state']
        
        # Get coordinates
        if state_name in INDIA_STATE_COORDINATES:
            coords = INDIA_STATE_COORDINATES[state_name]
            
            # Determine marker size and color based on metric
            metric_value = row[metric]
            
            # Normalize for visualization
            max_value = state_data[metric].max()
            min_value = state_data[metric].min()
            normalized = (metric_value - min_value) / (max_value - min_value) if max_value > min_value else 0.5
            
            # Color gradient from green to red
            if normalized < 0.33:
                color = 'green'
            elif normalized < 0.67:
                color = 'orange'
            else:
                color = 'red'
            
            # Marker size
            radius = 10 + (normalized * 30)
            
            # Create popup HTML
            popup_html = f"""
            <div style="font-family: Arial; width: 250px;">
                <h4 style="color: #1f77b4; margin-bottom: 10px;">{state_name}</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #f0f0f0;">
                        <td style="padding: 5px;"><b>Total Enrolments:</b></td>
                        <td style="padding: 5px; text-align: right;">{row['total_enrolment']:,.0f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px;"><b>Total Updates:</b></td>
                        <td style="padding: 5px; text-align: right;">{row['total_updates']:,.0f}</td>
                    </tr>
                    <tr style="background-color: #f0f0f0;">
                        <td style="padding: 5px;"><b>Demographic Updates:</b></td>
                        <td style="padding: 5px; text-align: right;">{row['total_demo_updates']:,.0f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px;"><b>Biometric Updates:</b></td>
                        <td style="padding: 5px; text-align: right;">{row['total_bio_updates']:,.0f}</td>
                    </tr>
                    <tr style="background-color: #f0f0f0;">
                        <td style="padding: 5px;"><b>Update Pressure Index:</b></td>
                        <td style="padding: 5px; text-align: right;">{row['update_pressure_index']:.4f}</td>
                    </tr>
                </table>
            </div>
            """
            
            # Add circle marker
            folium.CircleMarker(
                location=coords,
                radius=radius,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
                weight=2
            ).add_to(india_map)
            
            # Add state label
            folium.Marker(
                location=coords,
                icon=folium.DivIcon(html=f"""
                    <div style="font-size: 10px; font-weight: bold; color: black;">
                        {state_name[:3].upper()}
                    </div>
                """)
            ).add_to(india_map)
    
    # Add legend
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: auto;
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="font-weight: bold; margin-bottom: 10px;">Metric: {metric.replace('_', ' ').title()}</p>
        <p><span style="color: green;">●</span> Low (Bottom 33%)</p>
        <p><span style="color: orange;">●</span> Medium (33-67%)</p>
        <p><span style="color: red;">●</span> High (Top 33%)</p>
        <p style="font-size: 11px; margin-top: 10px; color: #666;">
            Click markers for detailed info
        </p>
    </div>
    """
    india_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(india_map)
    
    return india_map


def create_district_map(df, state_name, metric='total_enrolment'):
    """
    Create a detailed map for a specific state showing districts.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataframe
    state_name : str
        Name of the state to visualize
    metric : str
        Metric to visualize
        
    Returns:
    --------
    folium.Map
        Interactive district-level map
    """
    # Filter data for the state
    state_df = df[df['state'] == state_name].copy()
    
    if len(state_df) == 0:
        print(f"No data found for state: {state_name}")
        return None
    
    # Aggregate by district
    district_data = state_df.groupby(['state', 'district']).agg({
        'total_enrolment': 'sum',
        'total_updates': 'sum',
        'update_pressure_index': 'mean',
        'total_demo_updates': 'sum',
        'total_bio_updates': 'sum'
    }).reset_index()
    
    # Get state center
    state_center = INDIA_STATE_COORDINATES.get(state_name, [20.5937, 78.9629])
    
    # Create map
    district_map = folium.Map(
        location=state_center,
        zoom_start=7,
        tiles='OpenStreetMap'
    )
    
    # Create a feature group for districts
    district_group = folium.FeatureGroup(name='Districts')
    
    # Add markers for each district (distributed around state center)
    num_districts = len(district_data)
    for idx, row in district_data.iterrows():
        # Simple distribution of districts around state center
        angle = (idx / num_districts) * 2 * np.pi
        offset_lat = np.cos(angle) * 0.5
        offset_lon = np.sin(angle) * 0.5
        
        district_coords = [
            state_center[0] + offset_lat,
            state_center[1] + offset_lon
        ]
        
        # Popup content
        popup_html = f"""
        <div style="font-family: Arial; width: 250px;">
            <h4 style="color: #1f77b4;">{row['district']}</h4>
            <p><b>State:</b> {row['state']}</p>
            <hr>
            <p><b>Total Enrolments:</b> {row['total_enrolment']:,.0f}</p>
            <p><b>Total Updates:</b> {row['total_updates']:,.0f}</p>
            <p><b>Demographic Updates:</b> {row['total_demo_updates']:,.0f}</p>
            <p><b>Biometric Updates:</b> {row['total_bio_updates']:,.0f}</p>
            <p><b>Update Pressure Index:</b> {row['update_pressure_index']:.4f}</p>
        </div>
        """
        
        # Determine color based on metric
        metric_value = row[metric]
        max_val = district_data[metric].max()
        min_val = district_data[metric].min()
        norm = (metric_value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        
        color = 'red' if norm > 0.67 else ('orange' if norm > 0.33 else 'green')
        
        folium.CircleMarker(
            location=district_coords,
            radius=8 + (norm * 15),
            popup=folium.Popup(popup_html, max_width=300),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6
        ).add_to(district_group)
    
    district_group.add_to(district_map)
    
    # Add layer control
    folium.LayerControl().add_to(district_map)
    
    return district_map


if __name__ == "__main__":
    print("Geospatial visualization module loaded")
    print("Functions available:")
    print("  - create_india_enrollment_map()")
    print("  - create_district_map()")
