import os

svg_template = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1640 720" width="1640" height="720">
  <defs>
    <style>
      .bg {{ fill: #f8fafc; }}
      .box {{ rx: 8px; ry: 8px; stroke-width: 2px; }}
      .box-input {{ fill: #e0f2fe; stroke: #0284c7; }}
      .box-scene {{ fill: #dcfce7; stroke: #16a34a; }}
      .box-render {{ fill: #fef9c3; stroke: #ca8a04; }}
      .box-opt {{ fill: #fee2e2; stroke: #dc2626; }}
      .box-app {{ fill: #f3e8ff; stroke: #9333ea; }}
      
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
  <rect width="1640" height="720" class="bg" />

  <!-- Columns Backgrounds -->
  <rect x="40" y="40" width="280" height="640" class="col-bg" filter="url(#shadow)" />
  <rect x="360" y="40" width="280" height="640" class="col-bg" filter="url(#shadow)" />
  <rect x="680" y="40" width="280" height="640" class="col-bg" filter="url(#shadow)" />
  <rect x="1000" y="40" width="280" height="640" class="col-bg" filter="url(#shadow)" />
  <rect x="1320" y="40" width="280" height="640" class="col-bg" filter="url(#shadow)" />

  <!-- Column Titles -->
  <text x="180" y="80" class="title">1. Data Inputs</text>
  <text x="500" y="80" class="title">2. Core Representation</text>
  <text x="820" y="80" class="title">3. Rendering</text>
  <text x="1140" y="80" class="title">4. Optimization &amp; Loss</text>
  <text x="1460" y="80" class="title">5. Applications</text>

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

def arrow_curve_up(x1, y1, x2, y2, label=""):
    path = f"M {x1} {y1} C {(x1+x2)/2} {y1-40}, {(x1+x2)/2} {y2-40}, {x2} {y2}"
    mx = (x1+x2)/2
    my = min(y1, y2) - 20
    return f'''<path d="{path}" class="arrow" marker-end="url(#head)" />
    <rect x="{mx-50}" y="{my-10}" width="100" height="20" fill="#ffffff" rx="4"/>
    <text x="{mx}" y="{my+4}" class="arrow-label">{label}</text>
    '''

def arrow_back(x1, y1, x2, y2, label="", style="arrow-red"):
    y_down = y1 + 50
    path = f"M {x1} {y1} L {x1} {y_down} L {x2} {y_down} L {x2} {y2}"
    label_html = f'''
    <rect x="{(x1+x2)/2 - 80}" y="{y_down - 10}" width="160" height="20" fill="#ffffff" rx="4"/>
    <text x="{(x1+x2)/2}" y="{y_down + 4}" class="arrow-label">{label}</text>
    '''
    return f'''<path d="{path}" class="{style}" marker-end="url(#head-red)" />{label_html}'''

def arrow_loss_to_opt(y1, y2):
    # Route around the right side of the loss/opt column
    return f'<path d="M 1250 {y1} C 1300 {y1}, 1300 {y2}, 1250 {y2}" class="arrow" marker-end="url(#head)" />'

def arrow_tm_to_apps(x1, y1, x2, y2s, label=""):
    track_y = 660
    track_x = 1290 # Right before the Apps column (x=1350)
    gap_x = 660 # Gap between Column 2 and Column 3
    html = []
    
    # Main trunk
    trunk = f"M {x1} {y1} L {gap_x} {y1} L {gap_x} {track_y} L {track_x} {track_y}"
    html.append(f'<path d="{trunk}" class="arrow" />')
    
    # Branches to each app box
    for y2 in y2s:
        branch = f"M {track_x} {track_y} L {track_x} {y2} L {x2} {y2}"
        html.append(f'<path d="{branch}" class="arrow" marker-end="url(#head)" />')
        
    # Label on the bottom track
    html.append(f'''
    <rect x="{(gap_x+track_x)/2 - 60}" y="{track_y - 12}" width="120" height="24" fill="#ffffff" rx="4"/>
    <text x="{(gap_x+track_x)/2}" y="{track_y + 4}" class="arrow-label">{label}</text>
    ''')
    return "\n".join(html)

content = []
# Standard clean arrows
content.append(arrow(290, 340, 390, 340, "Init Mesh"))
content.append(arrow(290, 240, 390, 320, "Depth Priors", "arrow-dashed"))
content.append(arrow_straight(500, 305, 500, 175, "", "arrow-dashed"))
content.append(arrow_straight(500, 375, 500, 505, "", "arrow-dashed"))
content.append(arrow(290, 540, 710, 360, "Viewpoint"))
content.append(arrow_straight(610, 340, 710, 340, "Splat Params"))
content.append(arrow_straight(820, 375, 820, 405, ""))
content.append(arrow(930, 420, 1030, 160, "Render RGB"))
content.append(arrow_curve_up(290, 140, 1030, 140, "GT RGB"))
content.append(arrow(930, 440, 1030, 240, "Render Depth"))
content.append(arrow_straight(290, 240, 1030, 240, "GT Depth"))
content.append(arrow_back(1140, 575, 500, 575, "Backprop &amp; Topology Update"))

# Fixed Loss -> Opt arrows (routed on the right side)
content.append(arrow_loss_to_opt(140, 540)) # Photometric (center Y=140) to Opt (center Y=540)
content.append(arrow_loss_to_opt(240, 540)) # Depth Reg (center Y=240) to Opt

# Fixed Apps arrows (routed via a clean bottom track below CUDA)
content.append(arrow_tm_to_apps(610, 340, 1350, [240, 340, 440], "Trained Model"))

# Old Box placements (Perfectly centered in columns)
content.append(box(70, 105, 220, 70, "box-input", "RGB Images", "(Ground Truth)", "📸"))
content.append(box(70, 205, 220, 70, "box-input", "Depth Maps", "(Depth Anything V2)", "📏"))
content.append(box(70, 305, 220, 70, "box-input", "Sparse Point Cloud", "(COLMAP / Blender)", "☁️"))
content.append(box(70, 505, 220, 70, "box-input", "Camera Poses", "", "🎥"))

content.append(box(390, 105, 220, 70, "box-scene", "Properties", "Opacity, SH, Size, Scaling", "⚙️"))
content.append(box(390, 305, 220, 70, "box-scene", "TriangleModel", "(scene/triangle_model.py)", "🔺"))
content.append(box(390, 505, 220, 70, "box-scene", "Dynamic Topology", "Densify, Split, Delaunay", "🔄"))

content.append(box(710, 305, 220, 70, "box-render", "CUDA Rasterizer", "(diff-triangle-mesh)", "🖥️"))
content.append(box(710, 405, 220, 70, "box-render", "Render Outputs", "RGB, Depth, Normals, Masks", "🖼️"))

content.append(box(1030, 105, 220, 70, "box-opt", "Photometric Loss", "(L1 + SSIM)", "📉"))
content.append(box(1030, 205, 220, 70, "box-opt", "Depth Regularization", "(vertex_depth_loss_hr)", "📉"))
content.append(box(1030, 505, 220, 70, "box-opt", "Optimizer", "(SparseGaussianAdam)", "🚀"))

content.append(box(1350, 205, 220, 70, "box-app", "Novel View Synthesis", "(eval.py)", "✨"))
content.append(box(1350, 305, 220, 70, "box-app", "Mesh Extraction", "(mesh.py + Open3D)", "🕸️"))
content.append(box(1350, 405, 220, 70, "box-app", "Object Extraction", "(segmentation/ + SAM 2)", "🧩"))

os.makedirs('assets', exist_ok=True)
with open("assets/architecture.svg", "w") as f:
    f.write(svg_template.format(content="\n".join(content)))
print("Successfully generated fixed architecture diagram to assets/architecture.svg!")

