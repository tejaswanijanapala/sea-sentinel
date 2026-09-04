/**
 * Sea Sentinel: Interactive Sonar Waterfall Viewer
 * Renders acoustic waterfall scans with Port/Starboard channels, nadir line, and target bounding overlays.
 * Supports multi-mode inspection: Raw Scan, Enhanced (Lee+CLAHE), and Detections & Shadows.
 */

class WaterfallViewer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.targets = [];
    this.selectedTargetId = null;
    this.currentMode = "overlay"; // "raw" | "enhanced" | "overlay"

    this.rawImage = null;
    this.enhancedImage = null;
    this.annotatedImage = null;

    // Default acoustic waterfall canvas size
    this.canvas.width = 1200;
    this.canvas.height = 400;

    this._generateSyntheticWaterfall();
    this._initEvents();
  }

  _generateSyntheticWaterfall() {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const imgData = this.ctx.createImageData(w, h);
    const data = imgData.data;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        const distFromNadir = Math.abs(x - w / 2);

        // Nadir blind zone (dark acoustic void near center trackline)
        if (distFromNadir < 18) {
          const nadirDark = Math.floor(Math.random() * 18);
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

  setViewMode(mode) {
    this.currentMode = mode;
    this.render();
  }

  _isImageValid(img) {
    return Boolean(img && img.complete && img.naturalWidth > 0 && img.naturalHeight > 0);
  }

  loadSonarImages({ rawUrl, enhancedUrl, annotatedUrl }) {
    if (rawUrl) {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        this.rawImage = img;
        this.render();
      };
      img.onerror = () => {
        console.warn("Raw sonar image could not be decoded by browser:", rawUrl);
        this.rawImage = null;
        this.render();
      };
      img.src = rawUrl;
    } else {
      this.rawImage = null;
    }

    if (enhancedUrl) {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        this.enhancedImage = img;
        this.render();
      };
      img.onerror = () => {
        console.warn("Enhanced sonar image could not be decoded:", enhancedUrl);
        this.enhancedImage = null;
        this.render();
      };
      img.src = enhancedUrl;
    } else {
      this.enhancedImage = null;
    }

    if (annotatedUrl) {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        this.annotatedImage = img;
        this.render();
      };
      img.onerror = () => {
        console.warn("Annotated sonar image could not be decoded:", annotatedUrl);
        this.annotatedImage = null;
        this.render();
      };
      img.src = annotatedUrl;
    } else {
      this.annotatedImage = null;
    }
  }

  _getTargetCanvasCoords(t, w, h) {
    const norm = t.norm_bbox;
    if (norm && (norm.x2 > norm.x1)) {
      const x1 = norm.x1 * w;
      const y1 = norm.y1 * h;
      const x2 = norm.x2 * w;
      const y2 = norm.y2 * h;
      return { x1, y1, x2, y2, bw: Math.max(8, x2 - x1), bh: Math.max(8, y2 - y1) };
    }
    const bbox = t.pixel_bbox || {};
    const imgW = (t.image_dimensions && t.image_dimensions.width) || (this.rawImage ? this.rawImage.naturalWidth : w) || w;
    const imgH = (t.image_dimensions && t.image_dimensions.height) || (this.rawImage ? this.rawImage.naturalHeight : h) || h;
    const sx = w / imgW;
    const sy = h / imgH;
    const x1 = (bbox.x1 || 0) * sx;
    const y1 = (bbox.y1 || 0) * sy;
    const x2 = (bbox.x2 || (bbox.x1 + 80)) * sx;
    const y2 = (bbox.y2 || (bbox.y1 + 60)) * sy;
    return { x1, y1, x2, y2, bw: Math.max(8, x2 - x1), bh: Math.max(8, y2 - y1) };
  }

  render() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // 1. Draw Base Background (Real Sonar Image or Synthetic)
    let activeImg = null;
    if (this.currentMode === "raw" && this._isImageValid(this.rawImage)) {
      activeImg = this.rawImage;
    } else if (this.currentMode === "enhanced" && this._isImageValid(this.enhancedImage)) {
      activeImg = this.enhancedImage;
    } else if (this.currentMode === "overlay") {
      activeImg = this._isImageValid(this.annotatedImage) ? this.annotatedImage :
                  this._isImageValid(this.enhancedImage) ? this.enhancedImage :
                  this._isImageValid(this.rawImage) ? this.rawImage : null;
    }

    if (activeImg && this._isImageValid(activeImg)) {
      try {
        ctx.drawImage(activeImg, 0, 0, w, h);
        ctx.fillStyle = "rgba(0, 240, 255, 0.03)";
        ctx.fillRect(0, 0, w, h);
      } catch (err) {
        console.warn("Waterfall drawImage failed safely:", err);
        this._generateSyntheticWaterfall();
      }
    } else {
      this._generateSyntheticWaterfall();
    }

    // If in Raw or Enhanced pure mode without overlays, don't draw bounding boxes
    if (this.currentMode !== "overlay") {
      return;
    }

    const hasAnnotatedRaster = (activeImg === this.annotatedImage && this._isImageValid(this.annotatedImage));

    // 2. Draw Targets Bounding Boxes, Overlays & Selection Highlights
    this.targets.forEach(t => {
      const coords = this._getTargetCanvasCoords(t, w, h);
      const { x1, y1, bw, bh } = coords;

      const isSelected = (t.object_id === this.selectedTargetId);

      // Color coding by risk
      let color = "#00e676"; // LOW
      if (t.risk_score === "HIGH") color = "#ff1744";
      else if (t.risk_score === "MEDIUM") color = "#ffab00";

      ctx.save();

      if (hasAnnotatedRaster) {
        // The annotated raster already has the U-Net mask, contours, and base boxes rendered.
        // Draw interactive selection glow/brackets when selected:
        if (isSelected) {
          ctx.lineWidth = 3;
          ctx.strokeStyle = "#00f0ff";
          ctx.shadowColor = "#00f0ff";
          ctx.shadowBlur = 16;
          ctx.strokeRect(x1 - 2, y1 - 2, bw + 4, bh + 4);
          ctx.fillStyle = "rgba(0, 240, 255, 0.15)";
          ctx.fillRect(x1 - 2, y1 - 2, bw + 4, bh + 4);
        }
      } else {
        // Fallback vector overlay on top of raw/enhanced image
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.strokeStyle = color;

        if (isSelected) {
          ctx.shadowColor = color;
          ctx.shadowBlur = 14;
        }

        ctx.strokeRect(x1, y1, bw, bh);

        // Semi-transparent fill tint
        ctx.fillStyle = isSelected ? "rgba(0, 240, 255, 0.22)" : "rgba(0, 230, 118, 0.12)";
        ctx.fillRect(x1, y1, bw, bh);

        // Target Label Tag
        const confPct = Math.round((t.calibrated_confidence || t.confidence || 0) * 100);
        const cleanClass = (t.class || "debris").replace(/_/g, " ").toUpperCase();
        const label = `[${t.object_id}] ${cleanClass}: ${confPct}%`;
        ctx.font = "bold 11px 'JetBrains Mono', monospace";
        const textW = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(11, 21, 45, 0.9)";
        ctx.fillRect(x1, Math.max(0, y1 - 18), textW + 8, 16);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.strokeRect(x1, Math.max(0, y1 - 18), textW + 8, 16);
        ctx.fillStyle = color;
        ctx.fillText(label, x1 + 4, Math.max(12, y1 - 6));
      }

      ctx.restore();
    });
  }

  _initEvents() {
    const findHitTarget = (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const scaleX = this.canvas.width / rect.width;
      const scaleY = this.canvas.height / rect.height;
      const clickX = (e.clientX - rect.left) * scaleX;
      const clickY = (e.clientY - rect.top) * scaleY;

      return this.targets.find(t => {
        const coords = this._getTargetCanvasCoords(t, this.canvas.width, this.canvas.height);
        return clickX >= coords.x1 && clickX <= coords.x2 && clickY >= coords.y1 && clickY <= coords.y2;
      });
    };

    // Click selection (centers target on map)
    this.canvas.addEventListener('click', (e) => {
      const clicked = findHitTarget(e);
      if (clicked && window.app) {
        window.app.onTargetSelected(clicked.object_id, { fly: true, force: true });
      }
    });

    // Hover / Pointing out detection
    this.canvas.addEventListener('mousemove', (e) => {
      const hit = findHitTarget(e);
      if (hit) {
        this.canvas.style.cursor = 'pointer';
        if (window.app && window.app.selectedTargetId !== hit.object_id) {
          window.app.onTargetSelected(hit.object_id, { fly: false });
        }
      } else {
        this.canvas.style.cursor = 'default';
      }
    });
  }
}

window.WaterfallViewer = WaterfallViewer;
