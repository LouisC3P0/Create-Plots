import streamlit as st
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
        file_path = st.file_uploader(f"Upload the file for dataset {i+1}", type=["csv"])
        if file_path is not None:
            data = pd.read_csv(file_path)
            datasets.append(data)
    return datasets

def extract_multiple_columns_data(datasets, plot_type):
    column_data = []
    valid_labels = []
    for i, data in enumerate(datasets):
        x_var = st.text_input(f"Enter the X variable name for dataset {i+1}")
        y_var = st.text_input(f"Enter the Y variable name for dataset {i+1}")
        group_var = None
        if plot_type not in [3, 7]:  # No grouping for line plots and rose plots
            group_var = st.text_input(f"Enter the grouping variable (e.g., Region) for dataset {i+1}")
        
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
                        st.warning(f"Y variable '{y_var}' in dataset {i+1} for group '{group}' contains non-numeric data and will be skipped.")
            else:
                x_values = data[x_var].dropna().values
                y_values = data[y_var].dropna().values
                if np.issubdtype(y_values.dtype, np.number):
                    column_data.append((x_values, y_values, None))
                    valid_labels.append(y_var)
                else:
                    st.warning(f"Y variable '{y_var}' in dataset {i+1} contains non-numeric data and will be skipped.")
        else:
            st.warning(f"X variable '{x_var}', Y variable '{y_var}', or grouping variable '{group_var}' not found in dataset {i+1}")
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

# Streamlit application
def main():
    st.title("Data Plotter Application")

    plot_type = st.selectbox('Select Plot Type', 
                             ['Box Plot', 'Bar Plot', 'Line Plot', 'Violin + Strip Plot', 'Pseudo Color Plot', 'Spectrogram', 'Rose Plot'])

    input_color = None
    if plot_type in ['Box Plot', 'Bar Plot', 'Line Plot', 'Violin + Strip Plot', 'Rose Plot']:
        input_color = st.color_picker('Pick a Plot Color', '#00f900')

    num_datasets = st.number_input("Enter the number of datasets", min_value=1, max_value=10, value=1)
    datasets = load_datasets(num_datasets)

    plot_title = st.text_input('Plot Title', 'My Plot')
    x_label = st.text_input('X Label', 'X-axis')
    y_label = st.text_input('Y Label', 'Y-axis')
    legend_truefalse = st.selectbox('Include Legend?', ['Yes', 'No'])

    if st.button('Plot'):
        if plot_type in ['Box Plot', 'Bar Plot', 'Line Plot', 'Violin + Strip Plot']:
            column_data, valid_labels = extract_multiple_columns_data(datasets, plot_type)

            if column_data:
                fig, ax = plt.subplots()

                if plot_type == 'Box Plot':
                    notch = st.checkbox('Notched Box Plot')
                    flierprops = dict(marker='o', color='black', alpha=0.5, markersize=5)
                    ax.boxplot([data[1] for data in column_data], notch=notch, patch_artist=True, flierprops=flierprops, boxprops=dict(facecolor=input_color))

                elif plot_type == 'Bar Plot':
                    for i, (x_values, y_values, group) in enumerate(column_data):
                        y_mean = y_values.mean()
                        y_err = calculate_sem(y_values)
                        ax.bar(valid_labels[i], y_mean, yerr=y_err, alpha=0.7, capsize=10, color=input_color, label=valid_labels[i])

                elif plot_type == 'Line Plot':
                    for i, (x_values, y_values, group) in enumerate(column_data):
                        ax.plot(x_values, y_values, label=valid_labels[i], color=input_color)
                        ax.errorbar(x_values, y_values, yerr=calculate_sem(y_values), capsize=10, color=input_color)

                elif plot_type == 'Violin + Strip Plot':
                    for i, (x_values, y_values, group) in enumerate(column_data):
                        sns.violinplot(x=x_values, y=y_values, color=input_color, label=valid_labels[i])
                        sns.stripplot(x=x_values, y=y_values, color=input_color, legend=False)

                ax.set_title(plot_title)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                if legend_truefalse == 'Yes':
                    ax.legend()

                st.pyplot(fig)

            else:
                st.warning("No valid data to plot.")

        elif plot_type in ['Pseudo Color Plot', 'Spectrogram']:
            combined_data = None

            for i, data in enumerate(datasets):
                if combined_data is None:
                    combined_data = data.values
                else:
                    combined_data += data.values
            combined_data /= len(datasets)  # Averaging the datasets

            fig, ax = plt.subplots()

            if plot_type == 'Pseudo Color Plot':
                c = ax.pcolor(combined_data, cmap='jet')
                plt.colorbar(c, ax=ax)

            elif plot_type == 'Spectrogram':
                c = ax.imshow(combined_data, aspect='auto', cmap='jet')
                plt.colorbar(c, ax=ax)

            ax.set_title(plot_title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if legend_truefalse == 'Yes':
                ax.legend()

            st.pyplot(fig)

        elif plot_type == 'Rose Plot':
            if len(datasets) != 1:
                st.warning("Rose plot requires a single dataset.")
            else:
                fig, ax = create_rose_plot(datasets[0], input_color)
                ax.set_title(plot_title)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
