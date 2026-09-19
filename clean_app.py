import os
import re

def clean_app(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # The block to remove starts around "if (this.uploadedFile && window.apiService && window.apiService._runEdgeSimulationInference) {"
    # We can replace the fallback block with just a toast and a return/throw.
    
    # We will match the else block inside the catch
    pattern = r'\} else \{\s*// If an unexpected network or fetch error slipped through.*?\}\s*\}'
    
    replacement = '''} else {
        // Network or fetch error
        this.showToast({
          type: "error",
          title: "Inference Error",
          message: "Model inference unavailable. Cloud backend is unreachable."
        });
        if (statusPill && statusText) {
          statusPill.className = "status-pill error";
          statusText.textContent = "PIPELINE FAILED";
        }
      }'''
      
    # Using re.sub with DOTALL
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

clean_app('frontend/js/app.js')
clean_app('frontend/shared/js/app.js')
print("Clean successful")
