import os

svg_template = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1640 820" width="1640" height="820">
  <defs>
    <style>
      .bg {{ fill: #f8fafc; }}
      .box {{ rx: 8px; ry: 8px; stroke-width: 2px; }}
      .box-prep {{ fill: #e0f2fe; stroke: #0284c7; }}
      .box-train {{ fill: #fee2e2; stroke: #dc2626; }}
      .box-extract {{ fill: #fef9c3; stroke: #ca8a04; }}
      .box-eval {{ fill: #dcfce7; stroke: #16a34a; }}
      
      .col-bg {{ fill: #ffffff; stroke: #e2e8f0; stroke-width: 2px; rx: 12px; }}
      
      .title {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 20px; font-weight: bold; fill: #334155; text-anchor: middle; }}
      .text-main {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 15px; font-weight: bold; fill: #0f172a; text-anchor: middle; }}
      .text-sub {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px; fill: #475569; text-anchor: middle; }}
      
      .arrow {{ fill: none; stroke: #94a3b8; stroke-width: 3px; }}
      .arrow-dashed {{ fill: none; stroke: #94a3b8; stroke-width: 3px; stroke-dasharray: 6 6; }}
      .arrow-red {{ fill: none; stroke: #ef4444; stroke-width: 3px; stroke-dasharray: 6 6; }}
      .arrow-label {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px; font-weight: bold; fill: #475569; text-anchor: middle; }}
    </style>
    <marker id="head" orient="auto" markerWidth="6" markerHeight="8" refX="5.5" refY="4">
      <path d="M0,0 V8 L6,4 Z" fill="#94a3b8" />
    </marker>
    <marker id="head-red" orient="auto" markerWidth="6" markerHeight="8" refX="5.5" refY="4">
      <path d="M0,0 V8 L6,4 Z" fill="#ef4444" />
    </marker>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.05" />
    </filter>
  </defs>

  <!-- Background -->
  <rect width="1640" height="820" class="bg" />

  <!-- Columns Backgrounds -->
  <rect x="40" y="40" width="350" height="740" class="col-bg" filter="url(#shadow)" />
  <rect x="430" y="40" width="350" height="740" class="col-bg" filter="url(#shadow)" />
  <rect x="820" y="40" width="350" height="740" class="col-bg" filter="url(#shadow)" />
  <rect x="1210" y="40" width="350" height="740" class="col-bg" filter="url(#shadow)" />

  <!-- Column Titles -->
  <text x="215" y="80" class="title">1. Data Prep (Scripts 01-02)</text>
  <text x="605" y="80" class="title">2. Training (Script 03)</text>
  <text x="1000" y="80" class="title">3. Mesh Extract (Script 05)</text>
  <text x="1385" y="80" class="title">4. Eval &amp; Store (Script 06)</text>

  <!-- Nodes and Edges -->
  {content}
</svg>
"""

def box(x, y, w, h, cls, title, subtitle="", icon=""):
    cx = x + w/2
    cy = y + h/2
    sub_y = cy + 8 if subtitle else cy + 5
    title_y = cy - 8 if subtitle else cy + 5
    sub_html = f'<text x="{cx}" y="{sub_y+12}" class="text-sub">{subtitle}</text>' if subtitle else ""
    return f'''
  <g filter="url(#shadow)">
    <rect x="{x}" y="{y}" width="{w}" height="{h}" class="box {cls}" />
    <text x="{cx}" y="{title_y}" class="text-main">{icon} {title}</text>
    {sub_html}
  </g>'''

def arrow(x1, y1, x2, y2, label="", style="arrow"):
    mx = (x1 + x2) / 2
    path = f"M {x1} {y1} C {x1+50} {y1}, {x2-50} {y2}, {x2} {y2}"
    marker = "url(#head-red)" if "red" in style else "url(#head)"
    label_html = ""
    if label:
        label_html = f'''
        <rect x="{mx-50}" y="{(y1+y2)/2 - 10}" width="100" height="20" fill="#ffffff" rx="4"/>
        <text x="{mx}" y="{(y1+y2)/2 + 4}" class="arrow-label">{label}</text>
        '''
    return f'''<path d="{path}" class="{style}" marker-end="{marker}" />{label_html}'''

def arrow_straight(x1, y1, x2, y2, label="", style="arrow"):
    path = f"M {x1} {y1} L {x2} {y2}"
    marker = "url(#head-red)" if "red" in style else "url(#head)"
    label_html = ""
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        label_html = f'''
        <rect x="{mx-50}" y="{my - 10}" width="100" height="20" fill="#ffffff" rx="4"/>
        <text x="{mx}" y="{my + 4}" class="arrow-label">{label}</text>
        '''
    return f'''<path d="{path}" class="{style}" marker-end="{marker}" />{label_html}'''
    
def arrow_back(x1, y1, x2, y2, label="", style="arrow-red"):
    y_down = y1 + 50
    path = f"M {x1} {y1} L {x1} {y_down} L {x2} {y_down} L {x2} {y2}"
    label_html = f'''
    <rect x="{(x1+x2)/2 - 80}" y="{y_down - 10}" width="160" height="20" fill="#ffffff" rx="4"/>
    <text x="{(x1+x2)/2}" y="{y_down + 4}" class="arrow-label">{label}</text>
    '''
    return f'''<path d="{path}" class="{style}" marker-end="url(#head-red)" />{label_html}'''

content = []

# Col 1: Prep
content.append(box(75, 120, 280, 70, "box-prep", "Dataset (T&T Truck)", "COLMAP Point Cloud & Cameras", "📦"))
content.append(box(75, 240, 280, 70, "box-prep", "Depth Anything V2", "Generates Monocular Depth Priors", "📏"))
content.append(box(75, 360, 280, 70, "box-prep", "depth_params.json", "Scaling parameters", "⚙️"))
content.append(arrow_straight(215, 190, 215, 240, ""))
content.append(arrow_straight(215, 310, 215, 360, ""))

# Col 2: Train
content.append(box(465, 120, 280, 70, "box-train", "TriangleModel Init", "Pool of independent 2D splats", "🔺"))
content.append(box(465, 240, 280, 70, "box-train", "CUDA Rasterization", "Differentiable rendering", "🖥️"))
content.append(box(465, 360, 280, 70, "box-train", "Loss Computation", "L1, SSIM, Depth Reg", "📉"))
content.append(box(465, 480, 280, 70, "box-train", "SparseGaussianAdam", "Optimizes attributes (SH, size)", "🚀"))
content.append(box(465, 600, 280, 70, "box-train", "Topology Updates", "Delaunay Triangulation used here!", "🔄"))
content.append(box(440, 680, 330, 40, "box-train", "Keeps splat triangles conditioned during training", "", "💡"))

content.append(arrow_straight(465+140, 190, 465+140, 240, ""))
content.append(arrow_straight(465+140, 310, 465+140, 360, ""))
content.append(arrow_straight(465+140, 430, 465+140, 480, ""))
content.append(arrow_straight(465+140, 550, 465+140, 600, ""))
content.append(arrow_back(465+140, 720, 465+140, 205, "Next Iteration (Backprop)"))

# Col 1 -> Col 2
content.append(arrow(355, 155, 465, 155, "Init"))
content.append(arrow(355, 275, 465, 395, "Depth Prior"))

# Col 3: Extract
content.append(box(855, 240, 280, 70, "box-extract", "Extract Native Mesh", "06_extract_native.sh", "📸"))
content.append(box(855, 360, 280, 70, "box-extract", "create_ply.py", "Exports Delaunay-connected splats", "🧊"))
content.append(box(855, 480, 280, 70, "box-extract", "Opaque Connected Mesh", "Directly usable by game engines", "⛏️"))
content.append(box(855, 600, 280, 70, "box-extract", "native_mesh_30000.ply", "Final representation", "🕸️"))
content.append(box(830, 680, 330, 40, "box-extract", "Skips TSDF/Marching Cubes completely", "", "💡"))

content.append(arrow(465+280, 155, 855, 275, "Trained Splats (Triangle Soup)"))
content.append(arrow_straight(855+140, 310, 855+140, 360, ""))
content.append(arrow_straight(855+140, 430, 855+140, 480, ""))
content.append(arrow_straight(855+140, 550, 855+140, 600, ""))

# Col 4: Eval
content.append(box(1245, 600, 280, 70, "box-eval", "Mesh Stats Evaluation", "Count Vertices & Faces", "📏"))
content.append(box(1245, 720, 280, 70, "box-eval", "MeshSplatting.json", "Aggregate final results", "📊"))

content.append(arrow(1135, 635, 1245, 635, "Evaluate"))
content.append(arrow_straight(1245+140, 670, 1245+140, 720, ""))

os.makedirs('/data1/hemanth/mesh-splatting/my_expts/baseline_mesh_splatting_metrics', exist_ok=True)
with open("/data1/hemanth/mesh-splatting/my_expts/baseline_mesh_splatting_metrics/architecture.svg", "w") as f:
    f.write(svg_template.format(content="\\n".join(content)))
print("Successfully generated detailed architecture diagram to /data1/hemanth/mesh-splatting/my_expts/baseline_mesh_splatting_metrics/architecture.svg!")
