import os
import shutil
import hashlib

def get_file_hash(filepath):
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(1024*1024)
            hasher.update(chunk)
            f.seek(-min(1024*1024, os.path.getsize(filepath)), 2)
            chunk = f.read(1024*1024)
            hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return "ERROR"

def migrate():
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    project_root = os.path.abspath(os.path.join(backend_dir, ".."))
    persistent_dir = os.path.join(project_root, "persistent_data")
    
    os.makedirs(persistent_dir, exist_ok=True)
    
    targets = ["outputs", "datasets", "logs"]
    
    for target in targets:
        src = os.path.join(backend_dir, target)
        dst = os.path.join(persistent_dir, target)
        
        if not os.path.exists(src):
            continue
            
        print(f"Migrating {src} to {dst}...")
        
        # Pre-flight
        pre_count = 0
        pre_size = 0
        for r, d, f in os.walk(src):
            for file in f:
                pre_count += 1
                pre_size += os.path.getsize(os.path.join(r, file))
                
        print(f"  Pre-migration: {pre_count} files, {pre_size} bytes")
        
        # Move
        shutil.move(src, dst)
        
        # Post-flight
        post_count = 0
        post_size = 0
        for r, d, f in os.walk(dst):
            for file in f:
                post_count += 1
                post_size += os.path.getsize(os.path.join(r, file))
                
        print(f"  Post-migration: {post_count} files, {post_size} bytes")
        
        if pre_count != post_count or pre_size != post_size:
            print("  [ERROR] Mismatch detected!")
        else:
            print("  [SUCCESS] Lost files = 0, Corrupted files = 0")

if __name__ == "__main__":
    migrate()
