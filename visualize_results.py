"""
Visualize the model results from a csv file
Usage: python visualize_results.py <csv_file> <output_file>
"""

import csv
import json
import argparse
import io
import base64
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
        }}
        .sample {{
            padding-top: 0px;
            padding-bottom: 10px;
            border-top: 1px solid #ccc;
        }}
        .grid-container {{
            display: flex;
            gap: 50px;
            justify-content: center;
            align-items: center;
        }}
        .grid-item {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .grid {{
            display: grid;
            gap: 1px;
            margin: 5px 0;
            border: 1px solid #ccc;
            width: fit-content;
            padding: 5px;
        }}
        .grid div {{
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            border: 1px solid #eee;
        }}
        .hidden {{
            display: none;
        }}
        .button-container {{
            text-align: center;
            margin-bottom: 20px;
        }}
        button {{
            background-color: #010101;
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
        }}
        h3 {{
            padding: 10px;
        }}
    </style>
    <script>
        function filterSamples(filter) {{
            var samples = document.getElementsByClassName('sample');
            for (var i = 0; i < samples.length; i++) {{
                if (filter === 'all') {{
                    samples[i].classList.remove('hidden');
                }} else if (samples[i].classList.contains(filter)) {{
                    samples[i].classList.remove('hidden');
                }} else {{
                    samples[i].classList.add('hidden');
                }}
            }}
        }}
        function toggleGrid(gridType) {{
            var gridItems = document.getElementsByClassName('grid-item');
            for (var i = 0; i < gridItems.length; i++) {{
                if (gridItems[i].classList.contains(gridType)) {{
                    gridItems[i].classList.remove('hidden');
                }} else {{
                    gridItems[i].classList.add('hidden');
                }}
            }}
        }}
        function filterBySampleId() {{
            var sampleId = document.getElementById('sampleIdInput').value;
            var samples = document.getElementsByClassName('sample');
            for (var i = 0; i < samples.length; i++) {{
                if (samples[i].dataset.sampleId === sampleId) {{
                    samples[i].classList.remove('hidden');
                }} else {{
                    samples[i].classList.add('hidden');
                }}
            }}
        }}
        function sortSamplesBy(attribute) {{
            var samples = Array.from(document.getElementsByClassName('sample'));
            samples.sort(function(a, b) {{
                return parseFloat(b.dataset[attribute]) - parseFloat(a.dataset[attribute]);
            }});
            var container = document.getElementById('samples-container');
            container.innerHTML = '';
            samples.forEach(function(sample) {{
                container.appendChild(sample);
            }});
        }}
    </script>
</head>
<body>
    <h1 style="text-align: center;">{report_title}</h1>
    <div class="button-container">
        <button onclick="filterSamples('all')">Show All ({total_count})</button>
        <button onclick="filterSamples('correct')">Show Correct ({correct_count})</button>
        <button onclick="filterSamples('incorrect')">Show Incorrect ({incorrect_count})</button>
        <button onclick="toggleGrid('padded')">Show Padded</button>
        <button onclick="toggleGrid('unpadded')">Show Unpadded</button>
    </div>
    <div class="button-container">
        <input type="text" id="sampleIdInput" placeholder="Enter Sample ID">
        <button onclick="filterBySampleId()">Filter by Sample ID</button>
    </div>
    <div class="button-container">
        <button onclick="sortSamplesBy('loss')">Sort by Loss</button>
        <button onclick="sortSamplesBy('accuracy')">Sort by Accuracy</button>
        <button onclick="sortSamplesBy('foregroundAccuracy')">Sort by Foreground Accuracy</button>
        <button onclick="sortSamplesBy('backgroundAccuracy')">Sort by Background Accuracy</button>
    </div>
    <div id="samples-container">
        {content}
    </div>
</body>
</html>
"""


def generate_html(csv_file, output_file, report_title):
    filename = os.path.splitext(os.path.basename(csv_file))[0]
    print(f"Processing {filename}")

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    total_count = len(rows)
    correct_count = sum(1 for row in rows if row['is_correct'].lower() == 'true')
    incorrect_count = len(rows) - correct_count

    samples_html = ""
    with tqdm(total=len(rows), desc="Processed grids", unit="grid") as pbar:
        for row in rows:
            sample_html = f"<div class='sample ' \
                            data-sample-id='{row['sample_id']}' \
                            data-loss='{row['loss']}' \
                            data-accuracy='{row['accuracy']}' \
                            data-foreground-accuracy='{row['foreground_accuracy']}' \
                            data-background-accuracy='{row['background_accuracy']}'>"
            sample_html += f"<h3 style='text-align: center; margin-top: 0px;'>Sample ID: {row['sample_id']}</h3>"
            if 'loss' in row:
                sample_html += (
                    f"<h4 style='text-align: center; margin: 0 auto;'>"
                    f"<span style='margin-right: 20px;'>Focal Loss: {float(row['loss']):.3f}</span>"
                    f"<span>Accuracy (mIOU): {float(row['accuracy']):.3f}</span>"
                    f"</h4>"
                )
                sample_html += (
                    f"<h4 style='text-align: center; margin: 0 auto;'>"
                    f"<span style='margin-right: 20px;'>Foreground Accuracy: {float(row['foreground_accuracy']) * 100:.2f}%</span>"
                    f"<span>Background Accuracy: {float(row['background_accuracy']) * 100:.2f}%</span>"
                    f"</h4>"
                )
            sample_html += "<div class='grid-container'>"
            sample_html += render_grid_html(row['input'], "Input Grid", "unpadded")
            sample_html += render_grid_html(row['target'], "Target Grid", "unpadded")
            sample_html += render_grid_html(row['predicted'], "Predicted Grid", "unpadded")
            sample_html += render_grid_html(row['padded_input'], "Padded Input Grid", "padded hidden")
            sample_html += render_grid_html(row['padded_target'], "Padded Target Grid", "padded hidden")
            sample_html += render_grid_html(row['padded_predicted'], "Padded Predicted Grid", "padded hidden")
            sample_html += "</div></div>"
            samples_html += sample_html
            pbar.update(1)

    html_content = HTML_TEMPLATE.format(
        report_title=report_title,
        total_count=total_count,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        content=samples_html
    )

    with open(output_file, 'w') as file:
        file.write(html_content)
    print(f"HTML visualization written to {output_file}")


def render_grid_html(grid, label, grid_type):
    image_base64 = render_grid_image(grid)
    return f"""
    <div class='grid-item {grid_type}'>
        <h4>{label}</h4>
        <img src='{image_base64}' alt='{label}'>
    </div>
    """


def render_grid_image(grid):
    grid_data = json.loads(grid)
    rows = len(grid_data)
    cols = len(grid_data[0]) if rows > 0 else 0

    # Convert grid data to a numpy array
    grid_array = np.array(grid_data)

    pixel_size = 0.5  # Size of each pixel in the grid
    fig, ax = plt.subplots(figsize=(cols * pixel_size, rows * pixel_size))
    plt.axis('off')

    # Display the grid
    ax.matshow(grid_array)

    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Encode the image as base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML report from CSV file")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="Directory to save the output HTML file")
    parser.add_argument("-t", "--report_title", type=str, help="Title of the report")

    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir if args.output_dir else os.path.dirname(input_file)
    report_title = args.report_title if args.report_title else os.path.splitext(os.path.basename(input_file))[0]

    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}.html")

    generate_html(input_file, output_file, report_title)
