import os
import re
import shutil

MODULES = [
    "debris_density",
    "debris_detection",
    "debris_risk_scoring",
    "duplicate_detection",
    "natural_manmade_classification",
    "review_intelligence",
    "sonar_image_processing",
    "sonar_quality"
]

def consolidate_modules(base_dir):
    for mod in MODULES:
        mod_dir = os.path.join(base_dir, mod)
        if not os.path.isdir(mod_dir):
            print(f"Skipping {mod}, directory not found.")
            continue
            
        consolidated_content = f"\"\"\"{mod} module (consolidated)\"\"\"\n\n"
        
        # Collect all python files
        py_files = []
        for root, _, files in os.walk(mod_dir):
            for f in files:
                if f.endswith(".py") and f != "__init__.py":
                    py_files.append(os.path.join(root, f))
                    
        # Read and merge
        for py_file in py_files:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Remove imports pointing to itself (intra-module)
                # e.g., from backend.debris_density.services.density_service import ...
                pattern = r"^from backend\." + mod + r"\..*import.*\n?"
                content = re.sub(pattern, "", content, flags=re.MULTILINE)
                
                consolidated_content += f"\n# --- Extracted from {os.path.relpath(py_file, base_dir)} ---\n"
                consolidated_content += content + "\n"
                
        # Write to single file
        out_file = os.path.join(base_dir, f"{mod}.py")
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(consolidated_content)
        print(f"Consolidated {mod} into {mod}.py")

def update_global_imports(base_dir):
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".py") and f != "refactor.py":
                filepath = os.path.join(root, f)
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                original_content = content
                for mod in MODULES:
                    # Replace `from backend.debris_density.services.density_service import DebrisDensityService`
                    # with    `from backend.debris_density import DebrisDensityService`
                    
                    # Pattern matching "backend.module.something.something"
                    pattern1 = r"backend\." + mod + r"\.[a-zA-Z0-9_\.]+"
                    content = re.sub(pattern1, f"backend.{mod}", content)
                    
                    # Also catch cases without "backend." prefix
                    # e.g. `from debris_density.services.density_service`
                    pattern2 = r"from " + mod + r"\.[a-zA-Z0-9_\.]+"
                    content = re.sub(pattern2, f"from {mod}", content)

                if content != original_content:
                    with open(filepath, 'w', encoding='utf-8') as file:
                        file.write(content)
                    print(f"Updated imports in {os.path.relpath(filepath, base_dir)}")

def delete_old_dirs(base_dir):
    for mod in MODULES:
        mod_dir = os.path.join(base_dir, mod)
        if os.path.isdir(mod_dir):
            shutil.rmtree(mod_dir)
            print(f"Deleted old directory {mod}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    print("Starting consolidation...")
    consolidate_modules(base_dir)
    print("\nUpdating imports...")
    update_global_imports(base_dir)
    print("\nDeleting old directories...")
    delete_old_dirs(base_dir)
    print("\nDone.")
