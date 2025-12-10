import os
import uuid
import logging
import traceback
import shutil
from pathlib import Path
import types

import numpy as np
import torch
import einops
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from omegaconf import OmegaConf

# RollingDepth imports
from rollingdepth import (
    RollingDepthOutput,
    RollingDepthPipeline,
    get_video_fps,
)

# Configure Logging
logging.basicConfig(level=logging.INFO)

class RollingDepthServer:
    def __init__(self):
        # -------------------- Server Config --------------------
        self.work_root = "server_tmp"
        os.makedirs(self.work_root, exist_ok=True)
        self.checkpoint = "prs-eth/rollingdepth-v1-0"
        
        # Initialize Flask
        self.app = Flask(__name__)
        self.setup_routes()

        # -------------------- Model Initialization --------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading RollingDepth model on {self.device}...")
        
        # Use fp32 as default
        self.pipe: RollingDepthPipeline = RollingDepthPipeline.from_pretrained(
            self.checkpoint, torch_dtype=torch.float32
        )
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logging.info("xformers enabled")
        except ImportError:
            logging.warning("Run without xformers")

        self.pipe = self.pipe.to(self.device)
        
        # --- FIX: Register Hook to auto-move inputs to GPU ---
        self.register_gpu_hook()
        
        logging.info("Model loaded successfully.")

    def register_gpu_hook(self):
        """
        Registers a forward_pre_hook on the VAE encoder.
        This intercepts the input tensor right before it hits the GPU layers
        and ensures it is on the correct device.
        """
        def hook_fn(module, args):
            # args[0] is the input tensor (batch)
            x = args[0]
            # Check if input is a tensor and on the wrong device
            if torch.is_tensor(x) and x.device != self.device:
                x = x.to(self.device)
                return (x,) + args[1:]
            return args

        # The error happens in self.vae.encoder(batch)
        # So we attach the hook to the encoder module.
        self.pipe.vae.encoder.register_forward_pre_hook(hook_fn)
        logging.info("Applied GPU hook to VAE encoder.")

    def get_config_for_preset(self, preset_name):
        # Default Base Config
        args = OmegaConf.create(
            {
                "res": 768,
                "snippet_lengths": [3],
                "cap_dilation": True,
                "dtype": "fp32",
                "refine_snippet_len": 3,
                "refine_start_dilation": 6,
                "start_frame": 0,
                "frame_count": 0,
                "max_vae_bs": 4,
                "restore_res": False,
                "resample_method": "BILINEAR",
                "unload_snippet": False,
                "seed": None
            }
        )

        preset_args_dict = {
            "fast": OmegaConf.create({"dilations": [1, 25], "refine_step": 0, "dtype": "fp16"}),
            "fast1024": OmegaConf.create({"res": 1024, "dilations": [1, 25], "refine_step": 0, "dtype": "fp16"}),
            "full": OmegaConf.create({"res": 1024, "dilations": [1, 10, 25], "refine_step": 10}),
            "paper": OmegaConf.create({"dilations": [1, 10, 25], "cap_dilation": False, "dtype": "fp32", "refine_step": 10}),
        }

        if preset_name in preset_args_dict:
            logging.info(f"Applying preset: {preset_name}")
            args.update(preset_args_dict[preset_name])
        else:
            logging.warning(f"Preset '{preset_name}' not found. Using defaults.")
            if "dilations" not in args:
                args.dilations = [1, 10, 25]
                args.refine_step = 10

        return args

    def setup_routes(self):
        @self.app.route('/process_video', methods=['POST'])
        def process_video():
            request_id = uuid.uuid4().hex
            work_dir = Path(self.work_root) / request_id
            work_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                if 'video' not in request.files:
                    return jsonify({'error': 'No video file provided'}), 400
                
                video_file = request.files['video']
                if video_file.filename == '':
                    return jsonify({'error': 'No video filename'}), 400

                preset = request.form.get('preset', 'paper')

                filename = secure_filename(video_file.filename)
                input_video_path = work_dir / filename
                video_file.save(str(input_video_path))
                
                args = self.get_config_for_preset(preset)
                
                logging.info(f"[{request_id}] Starting inference on {filename} with preset {preset}")
                
                generator = None
                if args.seed is not None:
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(args.seed)

                with torch.no_grad():
                    pipe_out: RollingDepthOutput = self.pipe(
                        input_video_path=input_video_path,
                        start_frame=args.start_frame,
                        frame_count=args.frame_count,
                        processing_res=args.res,
                        resample_method=args.resample_method,
                        dilations=list(args.dilations),
                        cap_dilation=args.cap_dilation,
                        snippet_lengths=list(args.snippet_lengths),
                        init_infer_steps=[1],
                        strides=[1],
                        coalign_kwargs=None,
                        refine_step=args.refine_step,
                        refine_snippet_len=args.refine_snippet_len,
                        refine_start_dilation=args.refine_start_dilation,
                        generator=generator,
                        verbose=True,
                        max_vae_bs=args.max_vae_bs,
                        restore_res=args.restore_res,
                        unload_snippet=args.unload_snippet,
                    )

                depth_pred = pipe_out.depth_pred
                output_npy_filename = f"{input_video_path.stem}_depth.npy"
                output_npy_path = work_dir / output_npy_filename
                
                logging.info(f"[{request_id}] Saving npy to {output_npy_path}")
                np.save(output_npy_path, depth_pred.cpu().numpy().squeeze(1))

                return send_file(
                    output_npy_path,
                    mimetype='application/octet-stream',
                    as_attachment=True,
                    download_name=output_npy_filename
                )

            except Exception as e:
                logging.error(traceback.format_exc())
                return jsonify({'error': str(e)}), 500
            
            finally:
                if work_dir.exists():
                    try:
                        shutil.rmtree(work_dir)
                        logging.info(f"[{request_id}] Cleaned up temp directory.")
                    except Exception as e:
                        logging.error(f"Error cleaning up {work_dir}: {e}")

    def run(self):
        self.app.run(host='0.0.0.0', port=7000, debug=False)

if __name__ == '__main__':
    server = RollingDepthServer()
    server.run()
