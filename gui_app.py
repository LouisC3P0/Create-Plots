import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def calculate_sem(values):
    n = len(values)
    s = np.std(values, ddof=1)
    sem = s / np.sqrt(n)
    return sem

def load_datasets(num_datasets):
    datasets = []
    for i in range(num_datasets):
        while True:
            file_path = filedialog.askopenfilename(title=f"Select the file for dataset {i+1}")
            try:
                data = pd.read_csv(file_path)
                datasets.append(data)
                break
            except FileNotFoundError:
                messagebox.showerror("Error", f"File '{file_path}' not found. Please select a valid file.")
            except Exception as e:
                messagebox.showerror("Error", f"{e}. Please select a valid file.")
    return datasets

def extract_multiple_columns_data(datasets, plot_type):
    column_data = []
    valid_labels = []
    for i, data in enumerate(datasets):
        x_var = simpledialog.askstring("Input", f"Enter the X variable name for dataset {i+1}")
        y_var = simpledialog.askstring("Input", f"Enter the Y variable name for dataset {i+1}")
        group_var = None
        if plot_type not in [3, 7]:  # No grouping for line plots and rose plots
            group_var = simpledialog.askstring("Input", f"Enter the grouping variable (e.g., Region) for dataset {i+1}")
        
        if y_var in data.columns and x_var in data.columns and (group_var in data.columns if group_var else True):
            if group_var:
                for group in data[group_var].unique():
                    group_data = data[data[group_var] == group]
                    x_values = group_data[x_var].dropna().values
                    y_values = group_data[y_var].dropna().values
                    if np.issubdtype(y_values.dtype, np.number):
                        column_data.append((x_values, y_values, group))
                        valid_labels.append(f'{group}: {y_var}')
                    else:
                        messagebox.showwarning("Warning", f"Y variable '{y_var}' in dataset {i+1} for group '{group}' contains non-numeric data and will be skipped.")
            else:
                x_values = data[x_var].dropna().values
                y_values = data[y_var].dropna().values
                if np.issubdtype(y_values.dtype, np.number):
                    column_data.append((x_values, y_values, None))
                    valid_labels.append(y_var)
                else:
                    messagebox.showwarning("Warning", f"Y variable '{y_var}' in dataset {i+1} contains non-numeric data and will be skipped.")
        else:
            messagebox.showwarning("Warning", f"X variable '{x_var}', Y variable '{y_var}', or grouping variable '{group_var}' not found in dataset {i+1}")
    return column_data, valid_labels

def create_rose_plot(data, input_color):
    if 'angle' not in data.columns:
        raise ValueError("CSV file must contain an 'angle' column")
    
    if 'magnitude' not in data.columns:
        data['magnitude'] = 1

    data['angle_rad'] = np.deg2rad(data['angle'])
    
    num_bins = len(data)
    bins = np.linspace(0.0, 2 * np.pi, num_bins + 1)
    
    hist, bin_edges = np.histogram(data['angle_rad'], bins=bins, weights=data['magnitude'])
    
    width = 2 * np.pi / num_bins
    
    bottom = np.zeros_like(hist)

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    bars = ax.bar(bin_edges[:-1], hist, width=width, bottom=bottom, color=input_color, edgecolor=None, alpha=0.5)
    ax.grid(False)
    ax.set_yticklabels([])
    return fig, ax

class DataPlotterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Data Plotter GUI")
        self.geometry("400x300")
        self.plot_type = None
        self.input_color = None
        self.num_datasets = None
        self.datasets = []
        self.create_widgets()

    def create_widgets(self):
        tk.Button(self, text="Select Plot Type", command=self.select_plot_type).pack(pady=10)
        tk.Button(self, text="Load Datasets", command=self.load_datasets).pack(pady=10)
        tk.Button(self, text="Configure Plot", command=self.configure_plot).pack(pady=10)
        tk.Button(self, text="Create Plot", command=self.create_plot).pack(pady=10)

    def select_plot_type(self):
        self.plot_type_window = tk.Toplevel(self)
        self.plot_type_window.title("Select Plot Type")

        tk.Label(self.plot_type_window, text="Enter 1 for Box Plot, 2 for Bar Plot, 3 for Line Plot, 4 for Violin + Strip Plot, 5 for Pseudo Color Plot, 6 for Spectrogram, or 7 for Rose Plot").pack()
        self.plot_type_var = tk.StringVar()
        tk.Entry(self.plot_type_window, textvariable=self.plot_type_var).pack()

        tk.Button(self.plot_type_window, text="Next", command=self.save_plot_type).pack()

    def save_plot_type(self):
        self.plot_type = int(self.plot_type_var.get())
        self.plot_type_window.destroy()

    def load_datasets(self): 
        num_datasets = simpledialog.askinteger("Input", "Enter the number of datasets")
        self.datasets = load_datasets(num_datasets)

    def configure_plot(self):
        self.config_window = tk.Toplevel(self)
        self.config_window.title("Configure Plot")

        tk.Label(self.config_window, text="Plot Color:").pack()
        self.color_var = tk.StringVar()
        tk.Entry(self.config_window, textvariable=self.color_var).pack()

        tk.Label(self.config_window, text="Plot Title:").pack()
        self.title_var = tk.StringVar()
        tk.Entry(self.config_window, textvariable=self.title_var).pack()

        tk.Label(self.config_window, text="X Label:").pack()
        self.xlabel_var = tk.StringVar()
        tk.Entry(self.config_window, textvariable=self.xlabel_var).pack()

        tk.Label(self.config_window, text="Y Label:").pack()
        self.ylabel_var = tk.StringVar()
        tk.Entry(self.config_window, textvariable=self.ylabel_var).pack()

        tk.Label(self.config_window, text="Legend (True/False):").pack()
        self.legend_var = tk.StringVar()
        tk.Entry(self.config_window, textvariable=self.legend_var).pack()

        if self.plot_type == 1:  # Only show notch option for box plots
            tk.Label(self.config_window, text="Notch (True/False):").pack()
            self.notch_var = tk.StringVar()
            tk.Entry(self.config_window, textvariable=self.notch_var).pack()

        tk.Button(self.config_window, text="Next", command=self.save_config).pack()

    def save_config(self):
        self.input_color = self.color_var.get()
        self.plot_title = self.title_var.get()
        self.x_label = self.xlabel_var.get()
        self.y_label = self.ylabel_var.get()
        self.legend = self.legend_var.get().lower() in ['true', '1', 't', 'y', 'yes']
        if self.plot_type == 1:
            self.notch = self.notch_var.get().lower() in ['true', '1', 't', 'y', 'yes']
        else:
            self.notch = False
        self.config_window.destroy()

    def create_plot(self):
        if not self.datasets:
            messagebox.showerror("Error", "No datasets loaded.")
            return

        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)  # High-resolution figure

        # Set background to white and remove grid
        if self.plot_type in [1, 2, 3, 4]:
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            ax.grid(False)

            # Add black border around the plot area
            ax.spines['top'].set_edgecolor('black')
            ax.spines['top'].set_linewidth(2)
            ax.spines['right'].set_edgecolor('black')
            ax.spines['right'].set_linewidth(2)
            ax.spines['bottom'].set_edgecolor('black')
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_edgecolor('black')
            ax.spines['left'].set_linewidth(2)
        else:
            None

        if self.plot_type in [1, 2, 3, 4]:
            column_data, valid_labels = extract_multiple_columns_data(self.datasets, self.plot_type)

            if column_data:
                if self.plot_type == 1:  # Box Plot
                    flierprops = dict(marker='o', color='black', alpha=0.5, markersize=5)
                    ax.boxplot([data[1] for data in column_data], notch=self.notch, patch_artist=True, flierprops=flierprops, boxprops=dict(facecolor=self.input_color))

                elif self.plot_type == 2:  # Bar Plot
                    for i, (x_values, y_values, group) in enumerate(column_data):
                        y_mean = y_values.mean()
                        y_err = calculate_sem(y_values)
                        ax.bar(valid_labels[i], y_mean, yerr=y_err, alpha=0.7, capsize=10, color=self.input_color, label=valid_labels[i])

                elif self.plot_type == 3:  # Line Plot
                    for i, (x_values, y_values, group) in enumerate(column_data):
                        ax.plot(x_values, y_values, label=valid_labels[i], color=self.input_color)
                        ax.errorbar(x_values, y_values, yerr=calculate_sem(y_values), capsize=10, color=self.input_color)

                elif self.plot_type == 4:  # Violin + Strip Plot
                    for i, (x_values, y_values, group) in enumerate(column_data):
                        sns.violinplot(x=x_values, y=y_values, color=self.input_color, label=valid_labels[i])
                        sns.stripplot(x=x_values, y=y_values, color=self.input_color, legend=False)

            else:
                messagebox.showinfo("No Data", "No valid data to plot.")

        elif self.plot_type in [5, 6]:
            combined_data = None

            for i, data in enumerate(self.datasets):
                if combined_data is None:
                    combined_data = data.values
                else:
                    combined_data += data.values
            combined_data /= len(self.datasets)  # Averaging the datasets

            if self.plot_type == 5:  # Pseudo Color Plot
                c = ax.pcolor(combined_data, cmap='jet')
                plt.colorbar(c, ax=ax)

            elif self.plot_type == 6:  # Spectrogram
                c = ax.imshow(combined_data, aspect='auto', cmap='jet')
                plt.colorbar(c, ax=ax)

        elif self.plot_type == 7:  # Rose Plot
            if len(self.datasets) != 1:
                messagebox.showerror("Error", "Rose plot requires a single dataset.")
                return
            else:
                fig, ax = create_rose_plot(self.datasets[0], self.input_color)

        ax.set_title(self.plot_title)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        
        if self.legend:
            ax.legend()

        plt.show()

if __name__ == "__main__":
    app = DataPlotterApp()
    app.mainloop()
