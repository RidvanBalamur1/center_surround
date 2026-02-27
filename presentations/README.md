# Presentations

## Model Evolution Presentation

The file `model_evolution.md` contains a comprehensive presentation about the evolution of center-surround models from Classical Klindt to Dedicated ON/OFF Mixed.

### Viewing Options

#### Option 1: Marp (Recommended)
Install Marp CLI and convert to HTML/PDF:
```bash
# Install
npm install -g @marp-team/marp-cli

# Convert to HTML
marp model_evolution.md -o model_evolution.html

# Convert to PDF
marp model_evolution.md -o model_evolution.pdf

# Live preview
marp -p model_evolution.md
```

#### Option 2: VS Code Extension
1. Install "Marp for VS Code" extension
2. Open `model_evolution.md`
3. Click the Marp icon in the top right
4. Export to HTML/PDF/PPTX

#### Option 3: reveal.js
```bash
# Install reveal-md
npm install -g reveal-md

# Run presentation server
reveal-md model_evolution.md
```

#### Option 4: Pandoc
```bash
# Convert to PowerPoint
pandoc model_evolution.md -o model_evolution.pptx

# Convert to PDF via LaTeX beamer
pandoc model_evolution.md -t beamer -o model_evolution.pdf
```

#### Option 5: View as Markdown
Simply open in any markdown viewer (GitHub, VS Code, etc.)

### Content Overview

1. Problem introduction
2. Model architecture overview
3. Classical Klindt model
4. Per-Channel Masks model
5. N-Masks (Surround) model
6. ON/OFF model
7. ON/OFF Mixed (Shared) model
8. Dedicated ON/OFF Mixed model
9. Architecture comparison
10. Visualization techniques
11. Key takeaways
12. Future directions
