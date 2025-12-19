"""
Visualization script for simulation data.

Generates graphs from CSV files (states.csv, infections.csv, contacts.csv).

Usage:
    python DEMO\visualize_simulation.py
"""
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict, Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class SimulationVisualizer:
    def __init__(self, export_dir="DEMO/exports"):
        self.export_dir = export_dir
        self.states_df = None
        self.infections_df = None
        self.contacts_df = None
        self.load_data()

    def load_data(self):
        """Load CSV files into DataFrames."""
        states_path = os.path.join(self.export_dir, "states.csv")
        infections_path = os.path.join(self.export_dir, "infections.csv")
        contacts_path = os.path.join(self.export_dir, "contacts.csv")

        if os.path.exists(states_path):
            self.states_df = pd.read_csv(states_path)
            print(f"Loaded states: {len(self.states_df)} rows")
        else:
            print(f"Warning: {states_path} not found")

        if os.path.exists(infections_path):
            self.infections_df = pd.read_csv(infections_path)
            print(f"Loaded infections: {len(self.infections_df)} rows")
        else:
            print(f"Warning: {infections_path} not found")

        if os.path.exists(contacts_path):
            self.contacts_df = pd.read_csv(contacts_path)
            print(f"Loaded contacts: {len(self.contacts_df)} rows")
        else:
            print(f"Warning: {contacts_path} not found")

    def plot_epidemiological_curve(self):
        """Plot infected vs healthy over time (SIR-like curve)."""
        if self.states_df is None:
            print("No states data to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        
        self.states_df['healthy'] = self.states_df['total_people'] - self.states_df['infected']
        
        ax.plot(self.states_df['time'], self.states_df['healthy'], 
                label='Healthy', color='green', linewidth=2, marker='o', markersize=3)
        ax.plot(self.states_df['time'], self.states_df['infected'], 
                label='Infected', color='red', linewidth=2, marker='x', markersize=4)
        ax.plot(self.states_df['time'], self.states_df['total_people'], 
                label='Total Population', color='blue', linewidth=1.5, linestyle='--')
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Number of People', fontsize=12)
        ax.set_title('Epidemiological Curve - Virus Propagation', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'epidemiological_curve.png'), dpi=300)
        print("Saved: epidemiological_curve.png")
        plt.close()

    def plot_infection_rate(self):
        """Plot infection rate (new infections per time interval)."""
        if self.states_df is None:
            print("No states data to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate infection rate (derivative)
        self.states_df['infection_rate'] = self.states_df['infected'].diff() / self.states_df['time'].diff()
        self.states_df['infection_rate'].fillna(0, inplace=True)
        
        ax.bar(self.states_df['time'], self.states_df['infection_rate'], 
               color='darkred', alpha=0.7, width=0.5)
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('New Infections per Second', fontsize=12)
        ax.set_title('Infection Rate Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'infection_rate.png'), dpi=300)
        print("Saved: infection_rate.png")
        plt.close()

    def plot_infection_methods(self):
        """Plot distribution of infection methods."""
        if self.infections_df is None or len(self.infections_df) == 0:
            print("No infection data to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        
        method_counts = self.infections_df['method'].value_counts()
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        
        method_counts.plot(kind='bar', ax=ax, color=colors[:len(method_counts)], alpha=0.8)
        
        ax.set_xlabel('Infection Method', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of Infection Methods', fontsize=14, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(method_counts):
            ax.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'infection_methods.png'), dpi=300)
        print("Saved: infection_methods.png")
        plt.close()

    def plot_contact_distances(self):
        """Plot histogram of contact distances."""
        if self.contacts_df is None or len(self.contacts_df) == 0:
            print("No contact data to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(self.contacts_df['distance'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Distance (units)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Contact Distances', fontsize=14, fontweight='bold')
        ax.axvline(self.contacts_df['distance'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {self.contacts_df['distance'].mean():.2f}")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'contact_distances.png'), dpi=300)
        print("Saved: contact_distances.png")
        plt.close()

    def plot_infection_network(self, max_nodes=50):
        """Plot infection transmission network (who infected whom)."""
        if self.infections_df is None or len(self.infections_df) == 0:
            print("No infection data to plot")
            return

        # Create directed graph
        G = nx.DiGraph()
        
        # Sample data if too large
        df = self.infections_df
        if len(df) > max_nodes * 2:
            df = df.sample(n=max_nodes * 2, random_state=42)
        
        for _, row in df.iterrows():
            source = int(row['source']) if pd.notna(row['source']) else None
            target = int(row['target'])
            
            if source is not None:
                G.add_edge(source, target, method=row['method'])
            else:
                G.add_node(target)
        
        # Calculate node colors based on in-degree (how many people they infected)
        in_degrees = dict(G.in_degree())
        node_colors = [in_degrees.get(node, 0) for node in G.nodes()]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                              arrows=True, arrowsize=10, arrowstyle='->', 
                              alpha=0.5, width=1.5, connectionstyle='arc3,rad=0.1')
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                                       node_size=300, cmap='YlOrRd', 
                                       vmin=0, vmax=max(node_colors) if node_colors else 1)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color='black')
        
        ax.set_title('Infection Transmission Network\n(Node size/color = infectiousness)', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(nodes, ax=ax, label='Number of People Infected')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'infection_network.png'), dpi=300, bbox_inches='tight')
        print("Saved: infection_network.png")
        plt.close()

    def plot_contact_methods(self):
        """Plot distribution of contact methods."""
        if self.contacts_df is None or len(self.contacts_df) == 0:
            print("No contact data to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        
        method_counts = self.contacts_df['method'].value_counts()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        method_counts.plot(kind='bar', ax=ax, color=colors[:len(method_counts)], alpha=0.8)
        
        ax.set_xlabel('Contact Method', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of Contact Methods', fontsize=14, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(method_counts):
            ax.text(i, v + method_counts.max() * 0.02, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'contact_methods.png'), dpi=300)
        print("Saved: contact_methods.png")
        plt.close()

    def plot_most_contagious(self, top_n=10):
        """Plot most contagious people (infected the most others)."""
        if self.infections_df is None or len(self.infections_df) == 0:
            print("No infection data to plot")
            return

        # Count infections per person (source)
        contagious = self.infections_df['source'].value_counts().head(top_n)
        
        # Remove NaN (initial infections without source)
        contagious = contagious[contagious.index.notna()]
        
        if len(contagious) == 0:
            print("No source data available")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        
        contagious.plot(kind='barh', ax=ax, color='coral', alpha=0.8)
        
        ax.set_xlabel('Number of People Infected', fontsize=12)
        ax.set_ylabel('Person ID', fontsize=12)
        ax.set_title(f'Top {top_n} Most Contagious People', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(contagious):
            ax.text(v + 0.1, i, str(int(v)), va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'most_contagious.png'), dpi=300)
        print("Saved: most_contagious.png")
        plt.close()

    def plot_most_contagious_rooms(self, top_n=10):
        """Plot classrooms/rooms with most recorded infections."""
        if self.infections_df is None or len(self.infections_df) == 0:
            print("No infection data to plot")
            return

        # Try to detect the column that stores room information
        room_col = None
        for candidate in ["room", "location", "room_id", "classroom"]:
            if candidate in self.infections_df.columns:
                room_col = candidate
                break

        if room_col is None:
            print("No room column found in infections.csv (expected one of: room, location, room_id, classroom)")
            return

        room_counts = self.infections_df[room_col].value_counts().head(top_n)
        
        # Filter out 'unknown' rooms
        room_counts = room_counts[room_counts.index != 'unknown']

        if len(room_counts) == 0:
            print("No room data available to plot (only 'unknown' rooms)")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        room_counts.sort_values().plot(kind='barh', ax=ax, color='#6fa3ef', alpha=0.85)

        ax.set_xlabel('Number of Infections', fontsize=12)
        ax.set_ylabel('Room', fontsize=12)
        ax.set_title(f'Top {top_n} Most Contagious Rooms', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        for i, v in enumerate(room_counts.sort_values()):
            ax.text(v + 0.1, i, str(int(v)), va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'most_contagious_rooms.png'), dpi=300)
        print("Saved: most_contagious_rooms.png")
        plt.close()
    
    def plot_school_heatmap(self):
        """Plot heatmap of school showing infection hotspots by room."""
        if self.infections_df is None or len(self.infections_df) == 0:
            print("No infection data to plot heatmap")
            return
        
        # Check for room column
        room_col = None
        for candidate in ["room", "location", "room_id", "classroom"]:
            if candidate in self.infections_df.columns:
                room_col = candidate
                break
        
        if room_col is None:
            print("No room column found - cannot create heatmap")
            return
        
        # Get infection counts per room
        room_counts = self.infections_df[room_col].value_counts()
        room_counts = room_counts[room_counts.index != 'unknown']
        
        if len(room_counts) == 0:
            print("No valid room data for heatmap")
            return
        
        # Simple visualization - grid layout based on room names
        # Extract building/floor info from room codes like Q1-0007, Q2-1003, etc.
        room_data = []
        for room, count in room_counts.items():
            if '-' in str(room):
                parts = str(room).split('-')
                building = parts[0]  # Q1, Q2, Q3, Q4
                floor_room = parts[1] if len(parts) > 1 else '0000'
                floor = int(floor_room[0]) if floor_room else 0
                room_num = int(floor_room[1:]) if len(floor_room) > 1 else 0
                room_data.append({
                    'name': room,
                    'building': building,
                    'floor': floor,
                    'room_num': room_num,
                    'infections': count
                })
        
        if not room_data:
            print("Could not parse room structure for heatmap")
            return
        
        # Create pivot table for heatmap
        df_heatmap = pd.DataFrame(room_data)
        pivot = df_heatmap.pivot_table(
            values='infections',
            index='floor',
            columns='building',
            aggfunc='sum',
            fill_value=0
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.0f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Number of Infections'},
            linewidths=1,
            linecolor='gray',
            ax=ax
        )
        
        ax.set_xlabel('Building', fontsize=12, fontweight='bold')
        ax.set_ylabel('Floor', fontsize=12, fontweight='bold')
        ax.set_title('School Heatmap - Infection Hotspots by Building & Floor', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'school_heatmap.png'), dpi=300)
        print("Saved: school_heatmap.png")
        plt.close()

    def plot_total_population(self):
        """Plot total population over time."""
        if self.states_df is None:
            print("No states data to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.states_df['time'], self.states_df['total_people'], 
                color='#3498db', linewidth=2.5, marker='o', markersize=4, 
                markevery=max(1, len(self.states_df)//20))
        ax.fill_between(self.states_df['time'], self.states_df['total_people'], 
                        alpha=0.3, color='#3498db')
        
        ax.set_title('Total Population Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Number of People', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add max population annotation
        max_pop = self.states_df['total_people'].max()
        max_idx = self.states_df['total_people'].idxmax()
        max_time = self.states_df.loc[max_idx, 'time']
        ax.annotate(f'Max: {int(max_pop)} people', 
                   xy=(max_time, max_pop), 
                   xytext=(10, 10), 
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'total_population.png'), dpi=300)
        print("Saved: total_population.png")
        plt.close()
    
    def plot_infection_percentage(self):
        """Plot infection percentage over time."""
        if self.states_df is None:
            print("No states data to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate infection percentage, handling division by zero
        infection_pct = (self.states_df['infected'] / self.states_df['total_people'].replace(0, 1) * 100)
        infection_pct = infection_pct.fillna(0)
        
        ax.plot(self.states_df['time'], infection_pct, 
                color='#e74c3c', linewidth=2.5, marker='x', markersize=5,
                markevery=max(1, len(self.states_df)//20))
        ax.fill_between(self.states_df['time'], infection_pct, 
                        alpha=0.3, color='#e74c3c')
        
        ax.set_title('Infection Percentage Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Percentage Infected (%)', fontsize=12)
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add peak infection annotation
        if infection_pct.max() > 0:
            peak_pct = infection_pct.max()
            peak_idx = infection_pct.idxmax()
            peak_time = self.states_df.loc[peak_idx, 'time']
            ax.annotate(f'Peak: {peak_pct:.1f}%', 
                       xy=(peak_time, peak_pct), 
                       xytext=(10, -20), 
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'infection_percentage.png'), dpi=300)
        print("Saved: infection_percentage.png")
        plt.close()
    
    def plot_final_state_pie(self):
        """Plot final population state as pie chart."""
        if self.states_df is None:
            print("No states data to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        
        final_state = self.states_df.iloc[-1]
        healthy = final_state['total_people'] - final_state['infected']
        
        sizes = [healthy, final_state['infected']]
        labels = [f"Healthy\n{int(healthy)} people", f"Infected\n{int(final_state['infected'])} people"]
        colors_pie = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0.05)  # Slightly separate slices
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie, 
                                           autopct='%1.1f%%', startangle=90, 
                                           explode=explode,
                                           textprops={'fontsize': 12, 'fontweight': 'bold'},
                                           shadow=True)
        
        # Make percentage text white for better visibility
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
        
        ax.set_title('Final Population State', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'final_state_pie.png'), dpi=300)
        print("Saved: final_state_pie.png")
        plt.close()
    
    def plot_statistics_summary(self):
        """Plot statistics summary as text."""
        if self.states_df is None:
            print("No states data to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        final_state = self.states_df.iloc[-1]
        max_infected = self.states_df['infected'].max()
        attack_rate = (final_state['infected'] / final_state['total_people'] * 100) if final_state['total_people'] > 0 else 0
        
        stats_text = f"""
        ═══════════════════════════════════════════════
                   SIMULATION STATISTICS
        ═══════════════════════════════════════════════
        
        ⏱️  TEMPORAL DATA
           Duration:           {self.states_df['time'].max():.1f} seconds
           Simulation Start:   {self.states_df['time'].min():.1f} s
           Simulation End:     {self.states_df['time'].max():.1f} s
        
        👥  POPULATION DATA
           Max Population:     {int(self.states_df['total_people'].max())} people
           Final Population:   {int(final_state['total_people'])} people
           Average Population: {int(self.states_df['total_people'].mean())} people
        
        🦠  INFECTION DATA
           Max Infections:     {int(max_infected)} people
           Final Infections:   {int(final_state['infected'])} people
           Final Healthy:      {int(final_state['total_people'] - final_state['infected'])} people
        
        📊  KEY METRICS
           Attack Rate:        {attack_rate:.1f}%
           Peak Infection:     {(max_infected / self.states_df['total_people'].max() * 100):.1f}%
        
        ═══════════════════════════════════════════════
        """
        
        ax.text(0.5, 0.5, stats_text, fontsize=11, family='monospace',
               verticalalignment='center', horizontalalignment='center',
               bbox=dict(boxstyle='round,pad=1', facecolor='#ecf0f1', 
                        edgecolor='#34495e', linewidth=2, alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'statistics_summary.png'), dpi=300)
        print("Saved: statistics_summary.png")
        plt.close()

    def generate_all_plots(self):
        """Generate all visualizations."""
        print("\nGenerating visualizations...")
        print("-" * 50)
        
        self.plot_epidemiological_curve()
        self.plot_most_contagious()
        self.plot_most_contagious_rooms()
        self.plot_total_population()
        self.plot_infection_percentage()
        self.plot_final_state_pie()
        self.plot_statistics_summary()
        self.plot_school_heatmap()
        
        print("-" * 50)
        print(f"\nAll plots saved to: {self.export_dir}")


def main():
    # Try both possible export directories
    export_dirs = [
        os.path.join(os.getcwd(), "Versión final", "exports"),
        os.path.join(os.getcwd(), "DEMO", "exports")
    ]
    
    export_dir = None
    for d in export_dirs:
        if os.path.exists(d):
            export_dir = d
            break
    
    if export_dir is None:
        print(f"Error: No exports directory found")
        print("Tried: " + ", ".join(export_dirs))
        print("Run simulation_with_logger.py first to generate CSV files")
        return
    
    print(f"Loading data from: {export_dir}")
    visualizer = SimulationVisualizer(export_dir)
    visualizer.generate_all_plots()
    
    print("\n✅ Visualization complete! ")


if __name__ == "__main__":
    main()
