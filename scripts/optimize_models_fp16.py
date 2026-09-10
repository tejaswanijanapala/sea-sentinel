import os
import torch
import shutil

def inspect_and_optimize():
    print("==================================================")
    print("      SEA SENTINEL MODEL FP16 OPTIMIZER           ")
    print("==================================================")

    # 1. Inspect and optimize UNet
    unet_path = "models/unet/attention_unet_best.pt"
    if os.path.exists(unet_path):
        ckpt = torch.load(unet_path, map_location="cpu", weights_only=False)
        print("\n--- Inspecting Attention U-Net Checkpoint ---")
        if isinstance(ckpt, dict):
            print("Keys:", list(ckpt.keys()))
            state_dict = ckpt.get("model_state_dict", ckpt)
            total_params = 0
            float_params = 0
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    total_params += v.numel()
                    if v.is_floating_point():
                        float_params += v.numel()
            print(f"Total Parameters: {total_params:,} (Float: {float_params:,})")

            # Create clean FP16 model checkpoint (extract only model_state_dict in half precision, stripping optimizer states)
            fp16_state = {}
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    fp16_state[k] = v.half()
                else:
                    fp16_state[k] = v

            fp16_checkpoint = {
                "model_state_dict": fp16_state,
                "model_type": ckpt.get("model_type", "attention_unet"),
                "features": ckpt.get("features", [32, 64, 128, 256]),
                "precision": "FP16",
                "optimized_for": "edge_inference"
            }

            # Save clean FP16 checkpoint
            out_fp16 = "models/unet/attention_unet_best_fp16.pt"
            torch.save(fp16_checkpoint, out_fp16)
            
            # Also save to backend checkpoints
            b_out = "backend/models/checkpoints/unet/attention_unet_best_fp16.pt"
            os.makedirs(os.path.dirname(b_out), exist_ok=True)
            shutil.copyfile(out_fp16, b_out)

            orig_mb = os.path.getsize(unet_path) / (1024 * 1024)
            new_mb = os.path.getsize(out_fp16) / (1024 * 1024)
            saved_pct = ((orig_mb - new_mb) / orig_mb) * 100
            print(f"Original U-Net: {orig_mb:.2f} MB -> FP16 U-Net: {new_mb:.2f} MB [Saved {saved_pct:.1f}%!]")

    # 2. Inspect and optimize Autoencoder
    ae_path = "models/autoencoder/baseline_autoencoder.pt"
    if os.path.exists(ae_path):
        ckpt = torch.load(ae_path, map_location="cpu", weights_only=False)
        print("\n--- Inspecting Autoencoder Checkpoint ---")
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else {}
        fp16_state = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                fp16_state[k] = v.half()
            else:
                fp16_state[k] = v
        
        fp16_ae = {
            "model_state_dict": fp16_state,
            "precision": "FP16",
            "optimized_for": "edge_inference"
        }
        out_ae_fp16 = "models/autoencoder/baseline_autoencoder_fp16.pt"
        torch.save(fp16_ae, out_ae_fp16)
        b_ae = "backend/models/checkpoints/autoencoder/baseline_autoencoder_fp16.pt"
        os.makedirs(os.path.dirname(b_ae), exist_ok=True)
        shutil.copyfile(out_ae_fp16, b_ae)

        orig_mb = os.path.getsize(ae_path) / (1024 * 1024)
        new_mb = os.path.getsize(out_ae_fp16) / (1024 * 1024)
        saved_pct = ((orig_mb - new_mb) / orig_mb) * 100
        print(f"Original Autoencoder: {orig_mb:.2f} MB -> FP16 Autoencoder: {new_mb:.2f} MB [Saved {saved_pct:.1f}%!]")

    # 3. Inspect and optimize YOLO
    yolo_path = "models/yolo/best.pt"
    if os.path.exists(yolo_path):
        ckpt = torch.load(yolo_path, map_location="cpu", weights_only=False)
        print("\n--- Inspecting YOLO Checkpoint ---")
        if isinstance(ckpt, dict) and "model" in ckpt:
            model = ckpt["model"]
            if hasattr(model, "half"):
                model.half()
            
            # Save half precision YOLO checkpoint
            fp16_yolo = {
                "model": model,
                "epoch": -1,
                "precision": "FP16",
                "names": ckpt.get("names", {})
            }
            out_yolo_fp16 = "models/yolo/best_fp16.pt"
            torch.save(fp16_yolo, out_yolo_fp16)
            b_yolo = "backend/models/checkpoints/sih57_yolo_run/weights/best_fp16.pt"
            os.makedirs(os.path.dirname(b_yolo), exist_ok=True)
            shutil.copyfile(out_yolo_fp16, b_yolo)

            orig_mb = os.path.getsize(yolo_path) / (1024 * 1024)
            new_mb = os.path.getsize(out_yolo_fp16) / (1024 * 1024)
            saved_pct = ((orig_mb - new_mb) / orig_mb) * 100
            print(f"Original YOLO: {orig_mb:.2f} MB -> FP16 YOLO: {new_mb:.2f} MB [Saved {saved_pct:.1f}%!]")

if __name__ == "__main__":
    inspect_and_optimize()
