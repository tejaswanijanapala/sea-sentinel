import os

def clean_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Remove BENCHMARK_TARGETS
    # Already done by previous powershell command

    # 2. Find analyzeImage
    idx_analyze = content.find("async analyzeImage(")
    if idx_analyze == -1: return

    # 3. Find checkTrainingJob
    idx_check = content.find("async checkTrainingJob")
    if idx_check == -1:
        idx_check = content.find("async getTrainingMetrics")

    if idx_check == -1: return

    # Replacement for analyzeImage (and uploadFile is already broken so we just replace the whole chunk)
    # Let's rebuild the content
    new_analyze = """async analyzeImage(imagePath, rasterMeta = null, navLog = null, frameIdx = 1, mode = "balanced", fileRef = null) {
    try {
      const payload = {
        image_path: imagePath,
        raster_meta: rasterMeta,
        nav_log: navLog,
        frame_idx: frameIdx,
        mode: mode
      };
      const res = await fetch(`${this.baseUrl}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(45000)
      });
      if (!res.ok) {
        throw new Error(`Backend returned status ${res.status}`);
      }
      return await res.json();
    } catch (e) {
      console.warn("[SeaSentinel API] Cloud /api/analyze unavailable.", e);
      throw new Error("MODEL INFERENCE UNAVAILABLE");
    }
  }

  """

    # We also need to fix uploadFile before analyzeImage
    idx_upload = content.find("async uploadFile(")
    if idx_upload != -1:
        end_upload = content.find("async analyzeImage(")
        
        new_upload = """async uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${this.baseUrl}/api/upload`, {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(30000)
      });
      if (!res.ok) {
        throw new Error(`Upload failed with status ${res.status}`);
      }
      return await res.json();
    } catch (err) {
      throw new Error("MODEL INFERENCE UNAVAILABLE");
    }
  }

  """
        content = content[:idx_upload] + new_upload + new_analyze + content[idx_check:]
    else:
        content = content[:idx_analyze] + new_analyze + content[idx_check:]

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
clean_file('frontend/js/api.js')
clean_file('frontend/shared/js/api.js')
print("Clean successful")
