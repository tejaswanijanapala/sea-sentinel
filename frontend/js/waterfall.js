/**
 * Sea Sentinel: Interactive Sonar Waterfall Viewer
 * Renders acoustic waterfall scans with Port/Starboard channels, nadir line, and target bounding overlays.
 */

class WaterfallViewer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.targets = [];
    this.selectedTargetId = null;
    this.sonarImage = new Image();

    // Default acoustic waterfall canvas size
    this.canvas.width = 1200;
    this.canvas.height = 400;

    this._generateSyntheticWaterfall();
    this._initEvents();
  }

  _generateSyntheticWaterfall() {
    // Generates realistic textured seafloor backscatter (sand ripples, speckle noise)
    const w = this.canvas.width;
    const h = this.canvas.height;
    const imgData = this.ctx.createImageData(w, h);
    const data = imgData.data;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        const distFromNadir = Math.abs(x - w / 2);
        
        // Nadir blind zone (dark acoustic void near center)
        if (distFromNadir < 18) {
          const nadirDark = Math.floor(Math.random() * 20);
          data[idx] = nadirDark;
          data[idx + 1] = nadirDark + 5;
          data[idx + 2] = nadirDark + 10;
          data[idx + 3] = 255;
          continue;
        }

        // Ambient seafloor backscatter with grazing angle falloff
        let val = 75 + Math.sin(y * 0.08 + x * 0.02) * 15 + (Math.random() * 30 - 15);
        val = Math.max(25, Math.min(180, val));

        data[idx] = Math.floor(val * 0.85);       // Deep bronze / copper sonar tone
        data[idx + 1] = Math.floor(val * 0.95);
        data[idx + 2] = Math.floor(val * 1.15);
        data[idx + 3] = 255;
      }
    }
    this.ctx.putImageData(imgData, 0, 0);
  }

  setTargets(targets) {
    this.targets = targets || [];
    this.render();
  }

  selectTarget(targetId) {
    this.selectedTargetId = targetId;
    this.render();
  }

  render() {
    this._generateSyntheticWaterfall();
    const ctx = this.ctx;

    // Draw Targets Bounding Boxes
    this.targets.forEach(t => {
      const bbox = t.pixel_bbox || {};
      const x1 = bbox.x1 || 0;
      const y1 = bbox.y1 || 0;
      const w = (bbox.x2 || x1 + 80) - x1;
      const h = (bbox.y2 || y1 + 60) - y1;

      const isSelected = (t.object_id === this.selectedTargetId);

      // Color coding by risk
      let color = "#00e676"; // LOW
      if (t.risk_score === "HIGH") color = "#ff1744";
      else if (t.risk_score === "MEDIUM") color = "#ffab00";

      ctx.save();
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeStyle = color;

      // Glow effect if selected
      if (isSelected) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 12;
      }

      ctx.strokeRect(x1, y1, w, h);

      // Background fill tint
      ctx.fillStyle = isSelected ? "rgba(0, 240, 255, 0.15)" : "rgba(0, 0, 0, 0.3)";
      ctx.fillRect(x1, y1, w, h);

      // Target Label
      const confPct = Math.round((t.calibrated_confidence || t.confidence || 0) * 100);
      const label = `${t.object_id} [${confPct}%]`;
      ctx.font = "bold 11px 'JetBrains Mono', monospace";
      ctx.fillStyle = color;
      ctx.fillText(label, x1, Math.max(14, y1 - 4));

      ctx.restore();
    });
  }

  _initEvents() {
    this.canvas.addEventListener('click', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const scaleX = this.canvas.width / rect.width;
      const scaleY = this.canvas.height / rect.height;
      const clickX = (e.clientX - rect.left) * scaleX;
      const clickY = (e.clientY - rect.top) * scaleY;

      // Find clicked target
      const clicked = this.targets.find(t => {
        const b = t.pixel_bbox || {};
        return clickX >= b.x1 && clickX <= b.x2 && clickY >= b.y1 && clickY <= b.y2;
      });

      if (clicked && window.app) {
        window.app.onTargetSelected(clicked.object_id);
      }
    });
  }
}

window.WaterfallViewer = WaterfallViewer;
