## Advanced Configuration Guide

This guide documents all configuration knobs used by the SDG workflow across CARLA ground-truth generation, augmentation/orchestration, and deployment/runtime. It explains each field, types, defaults, constraints, and recommendations for first-time success and advanced tuning.

### Contents
- Environment and deployment variables
- Notebook runtime variables
- CARLA ground-truth configuration (JSON + camera YAML)
- Augmentation configuration (YAML)
- Validation rules and type constraints
- Recommendations and best practices
- Heterogeneous deployment and endpoints

---

## Environment and deployment variables
Source: `deploy/compose/env.example` (copied to `deploy/compose/env` for local edits)

- NIM_HOST: Hostname or IP where the NIM services (VLM/LLM/Cosmos-Transfer) run.
  - Type: string; Default: localhost
  - Recommendation: In heterogeneous deployments set to the NIM node IP.
- LOCAL_NIM_CACHE: Local cache directory for NIM. Default: ~/.cache/nim
- VLM_IMAGE: NIM image for Cosmos-Reason1. Example: nvcr.io/nim/nvidia/cosmos-reason1-7b:1.4.0
- VLM_PORT: Port for VLM endpoint. Default: 8001
- VLM_ENDPOINT: Derived as $NIM_HOST:$VLM_PORT
- VLM_GPU_ID, VLM_GPU_COUNT: GPU selection and count for VLM container
- LLM_IMAGE: NIM image for Nemotron model. Example: nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2:1
- LLM_PORT: Port for LLM endpoint. Default: 8002
- LLM_ENDPOINT: Derived as $NIM_HOST:$LLM_PORT
- LLM_GPU_ID, LLM_GPU_COUNT: GPU selection and count for LLM container
- TRANSFER_GRADIO_IMAGE: Cosmos-Transfer Gradio image tag. Example: cosmos-transfer2_5-gradio:v1.3.0
- TRANSFER_GRADIO_PORT: Cosmos-Transfer port. Default: 8080
- TRANSFER_ENDPOINT: http://$NIM_HOST:$TRANSFER_GRADIO_PORT
- TRANSFER_GPU_ID, TRANSFER_GPU_COUNT: GPU selection and count for Cosmos-Transfer
- TRANSFER_MODEL_NAME: Which Cosmos-Transfer model to start (service default is edge)
- HF_HOME: Hugging Face cache directory; default ~/.cache/huggingface
- NOTEBOOK_IMAGE: Workbench image. Default: smart-city-sdg-workbench:latest
- CARLA_SERVER_IMAGE: CARLA 0.9.16 image
- CARLA_HOST, CARLA_PORT: CARLA RPC host/port (default 2000)
- CARLA_STREAM_PORT_UDP, CARLA_STREAM_PORT_TCP: Playback stream ports
- CARLA_GPU_ID: GPU index for CARLA container
- NOTEBOOK_PORT: Jupyter port; default 8888
- NOTEBOOK_WORKDIR: Workbench starting directory (relative to container)
- NGC_API_KEY: Required to pull NIM images and checkpoints from `nvcr.io` and NGC.
- HF_TOKEN: Required to access Cosmos-Transfer2.5 checkpoints

Notes
- In homogeneous deployments, NIM_HOST=localhost.
- In heterogeneous deployments, set NIM_HOST on the Workbench node to the NIM node’s IP.
- Health checks:
  - export NIM_HOST=<nim_ip>; HOST=${NIM_HOST:-localhost}
  - curl http://$HOST:8001/v1/health/ready  # VLM
  - curl http://$HOST:8002/v1/health/ready  # LLM
  - Cosmos-Transfer service (optional browser verification): http://$HOST:8080

---

## Notebook runtime variables
Source: `notebooks/carla_synthetic_data_generation.ipynb` (USER-EDITABLE VARIABLES cell)

- BASE_INPUT_DIR: Constructed as $CARLA_OUTPUT_DIR/$RUN_ID; path to Stage-1 outputs.
- BASE_OUTPUT_DIR: Constructed as $COSMOS_OUTPUT_DIR/$RUN_ID; path for Stage-2 outputs.
- CONFIG_FILE_PATH: Augmentation YAML config path. Default: /workspace/modules/augmentation/configs/config_carla.yaml
- NUM_AUGMENTATIONS: Integer count per scenario.
- Endpoints resolved from env:
  - VLM_URL = http://$NIM_HOST:$VLM_PORT/v1
  - LLM_URL = http://$NIM_HOST:$LLM_PORT/v1
  - COSMOS_URL = http://$NIM_HOST:$TRANSFER_GRADIO_PORT/
- The notebook reads the default YAML (CONFIG_FILE_PATH), prints it, then writes a derived config by injecting data, endpoints, and selections.

Recommendations
- Ensure RUN_ID is unique per experiment to avoid overwriting outputs.
- Keep NUM_AUGMENTATIONS small initially (e.g., 1–2) to validate the pipeline end-to-end before scaling.

---

## CARLA ground-truth configuration
Stage-1 uses a JSON config (passed to `modules/carla-ground-truth-generation/main.py`) plus a camera YAML. Example files in `modules/carla-ground-truth-generation/config/`.

### JSON config fields (examples: `wrong_way.json`, `collision.json`)
- host: CARLA host (string). Default usually localhost
- port: CARLA RPC port (int). Default 2000
- timeout: Connection timeout seconds (float/int)
- camera_config: Path to camera YAML (string). Required
- recorder_filename: Path to CARLA .log recording (string). Required
- start_time: Seconds from which to start playback (float). Optional
- duration: Playback duration seconds (float). Optional
- time_factor: Speed multiplier for playback (float). Optional
- output_dir: Output directory for ground-truth and derived videos (string). Required
- generate_videos: Generate videos for rgb/depth/segmentation etc. (bool). Default: true
- limit_distance: Optional distance threshold for processing (float)
- area_threshold: Optional pixel/area threshold (float)
- class_filter_config: Optional YAML to filter semantic classes for edges (string)
- xodr_path: Optional OpenDRIVE map path (string)
- camera_follow_actor: Follow a specific actor (int/ID or descriptor)
- ignore_hero: Whether to ignore the hero vehicle (bool)
- move_spectator: Move spectator camera during playback (bool)
- detect_collisions: Whether to detect collisions (bool)
- collision_actor_ids: List of actor IDs to monitor for collisions (list[int])

Recommendations
- Start with `start_time` and short `duration` during validation runs.
- Set `generate_videos=true` initially to visually verify outputs.
- Use `class_filter_config` when you want masked edges; see below.

### Camera YAML schema (example: `third_person_camera.yaml`)
Array of sensors (the loader accepts both a top-level list or a dict with a `sensors` key wrapping the list). Each item:
- sensor: One of rgb, depth, semantic_segmentation, instance_segmentation, normals
- attributes:
  - image_size_x, image_size_y: Resolution (ints)
  - fov: Field of view (int/float)
- transform:
  - location: x, y, z
  - rotation: roll, pitch, yaw

Recommendations
- Keep all sensors aligned (same transform) to simplify downstream processing.
- 1080p with fov 90–110 is a reasonable starting range; adjust per workload and GPU.

### Class filter YAML (example: `filter_semantic_classes.yaml`)
- canny_classes: Array of BGR triplets for classes to include in masked-edge generation.
- Recommendation: Begin with provided defaults; expand to additional classes as needed.

---

### External examples (Inverted AI Metropolis)
The following reference configs align with this workflow’s inputs and are supported:
- Scenario JSON with timing and collision actors:
  - Fields: `start_time` (seconds), `duration` (seconds), `collision_actor_ids` (list[int])
  - Optional metadata like `source_scenario` may be present in external examples and is ignored by this loader.
  - Example: [`1.json`](https://raw.githubusercontent.com/inverted-ai/metropolis/master/examples/1.json)
- Camera YAML with a `sensors` wrapper (equivalent to a top-level list):
  - Example: [`1.yaml`](https://raw.githubusercontent.com/inverted-ai/metropolis/master/examples/1.yaml)
- Semantic class filter for masked edges (BGR Cityscapes palette):
  - Example: [`filter_semantic_classes.yaml`](https://raw.githubusercontent.com/inverted-ai/metropolis/master/examples/filter_semantic_classes.yaml)

Notes
- The CARLA loader automatically accepts camera YAML files either as a top-level list or with a `sensors:` wrapper.
- `collision_actor_ids` is supported by the loader. `source_scenario` is treated as metadata and not consumed.

---

## Augmentation configuration (YAML)
Source: `modules/augmentation/configs/config_carla.yaml`. This config is read and then augmented by the notebook with `data` and `endpoints`.

Top-level sections
- data: List of samples. Each sample has:
  - inputs:
    - rgb: Path to input video (string)
    - controls: Optional dict with per-modality control videos
      - edge, depth, seg: Paths to control videos (strings)
  - output:
    - video: Path to write stylized video (string)
    - caption: Path to write selected/polished prompt (string)
    - metadata: Path to write JSON metadata (string)
- endpoints: Service endpoints and models
  - vlm: { url, model }
  - llm: { url, model }
  - cosmos: { url, model }
  - Environment overrides (optional):
    - VLM_ENDPOINT_URL, VLM_ENDPOINT_MODEL
    - LLM_ENDPOINT_URL, LLM_ENDPOINT_MODEL
    - COSMOS_ENDPOINT_URL
- video_creation:
  - simulator: "carla" (placeholder for future video building)
- video_captioning:
  - system_prompt: String (required when video_captioning enabled)
  - user_prompt: String (required when enabled)
  - parameters:
    - retry: int (default 0)
    - temperature: float (default 0.6)
    - top_p: float (default 0.95)
    - frequency_penalty: float (default 1.05)
    - max_tokens: int (default 4096)
    - stream: bool (default false)
- template_generation:
  - system_prompt_file: Path to prompt file
  - parameters:
    - retry: int (default 1)
    - retry_policy: "default" | "check_vars"
    - temperature: float (default 0.0)
    - top_p: float (default 0.95)
    - max_tokens: int (default 4096)
    - frequency_penalty: float (default 0.0)
    - presence_penalty: float (default 0.0)
    - stream: bool (default true)
    - polish_temperature: float (default 0.8)
  - variables:
    - weather_condition: list[str]
    - lighting_condition: list[str]
    - road_condition: list[str]
- prompt_generation:
  - seed: int | null
  - save_prompts: bool (optional)
- cosmos:
  - executor_type: "gradio" | "nim" (current executor is gradio)
  - model_version: "ct1" | "ct2.5" | "ct25" (ct25 becomes ct2.5 internally)
  - configuration: Path to a TOML/aux config (if used by server)
  - parameters:
    - sigma: int (ct1 uses int; ct2.5 optionally uses string `sigma_max`)
    - seed: int | null (auto-set to current time if missing)
    - guidance: int
    - num_steps: int
    - inference_name: str (name for server run)
    - modalities: list[str] (e.g., ["edge"] or ["edge","depth","seg"])
    - weights: map[str -> float] (must include each modality key)
    - positive_prompt: str
    - negative_prompt: str
- logging:
  - enabled: bool
  - level: "INFO" | "DEBUG" | "WARNING" | "ERROR"

---

## Validation rules and type constraints
Source: `modules/augmentation/modules/utils/common.py::validate_config_structure`

Required structure and fields
- data must be a non-empty list; each sample must have inputs and output.
  - inputs.rgb is required.
  - output must contain video, caption, metadata.
- If any of video_captioning, template_generation, cosmos is enabled:
  - endpoints must exist.
- video_captioning enabled:
  - system_prompt, user_prompt, parameters required.
  - endpoints.vlm required.
- template_generation enabled:
  - parameters and variables required.
  - endpoints.llm required.
- cosmos enabled:
  - parameters required and must include sigma, seed, guidance, num_steps.
  - parameters.modalities: non-empty list
  - parameters.weights: must include a key for each modality in modalities
  - parameters.positive_prompt, parameters.negative_prompt required
  - endpoints.cosmos required
- prompt_generation requires template_generation.

Type expectations (strictly enforced in constructors)
- VLMSceneCaptioning: retry:int, temperature:float, top_p:float, frequency_penalty:float, max_tokens:int, stream:bool, system_prompt:str, user_prompt:str, endpoint:str, model:str
- LLMTemplateGenerator: system_prompt:str, retry:int, retry_policy:str, temperature:float, top_p:float, frequency_penalty:float, presence_penalty:float, max_tokens:int, stream:bool, endpoint:str, model:str, polish_temperature:float
- GradioCosmosExecutor: endpoint:str, sigma:int, seed:int, guidance:int, num_steps:int, modalities:list[str], weights:dict[str,float], positive_prompt:str, negative_prompt:str, inference_name:str, model_version:Optional[str]

---

## Recommendations and best practices
- Start small:
  - Set NUM_AUGMENTATIONS=1–2 and a short CARLA duration (e.g., 5s).
  - Use a single modality (edge) first; add depth/seg after validation.
- Cosmos parameters (good starting points):
  - sigma: 90, guidance: 3, num_steps: 35
  - modalities: ["edge"]; weights.edge: 1.0
  - Provide succinct, concrete positive_prompt; avoid lengthy negatives.
- Tokens and temperature:
  - For captioning and template generation, keep max_tokens ≤ 4096.
  - Use temperature 0.0 for template JSON; higher (0.6–0.8) for polishing.
- Seeds: If unset, seeds are auto-populated for reproducibility. Set explicit seeds for deterministic runs.
- Endpoints:
  - Use environment overrides to point at hosted endpoints if needed:
    - VLM_ENDPOINT_URL/MODEL, LLM_ENDPOINT_URL/MODEL, COSMOS_ENDPOINT_URL
  - In heterogeneous setups ensure NIM_HOST resolves and ports are reachable.
- Storage and S3 (optional):
  - If using S3 via `omni.client`, set AWS_* vars and bucket configuration.
- API keys:
  - BUILD_NVIDIA_API_KEY is required by LLMTemplateGenerator to call the LLM endpoint.

---

## Heterogeneous deployment and endpoints
- Set NIM_HOST on the Workbench node to the IP of the NIM node.
- Health:
  - HOST=${NIM_HOST:-localhost}
  - curl http://$HOST:8001/v1/health/ready  # VLM
  - curl http://$HOST:8002/v1/health/ready  # LLM
- Cosmos-Transfer browser UI (optional): http://$HOST:8080
- Notebook endpoint construction:
  - VLM_URL = http://$NIM_HOST:$VLM_PORT/v1
  - LLM_URL = http://$NIM_HOST:$LLM_PORT/v1
  - COSMOS_URL = http://$NIM_HOST:$TRANSFER_GRADIO_PORT/

---

## Minimal examples

CARLA JSON (excerpt)
```json
{ "camera_config": "config/scene1/cam1.yaml",
  "recorder_filename": "example_data/.../output_log.log",
  "output_dir": "output/.../cam1",
  "start_time": 5.0,
  "duration": 5.0,
  "generate_videos": true }
```

Augmentation YAML (key sections)
```yaml
data:
  - inputs:
      rgb: /path/to/rgb.mp4
      controls: { edge: /path/to/edge.mp4 }
    output:
      video: /path/to/output.mp4
      caption: /path/to/output.txt
      metadata: /path/to/output.json

endpoints:
  vlm: { url: "http://<host>:8001/v1", model: "nvidia/cosmos-reason1-7b" }
  llm: { url: "http://<host>:8002/v1", model: "nvidia/nvidia-nemotron-nano-9b-v2" }
  cosmos: { url: "http://<host>:8080/", model: "Cosmos-Transfer2.5-2B" }

video_captioning:
  system_prompt: "..."
  user_prompt: "..."
  parameters: { retry: 0, temperature: 0.6, top_p: 0.95, max_tokens: 4096, stream: false }

template_generation:
  system_prompt_file: "prompts/carla_template_generation_system_prompt.txt"
  parameters: { retry: 1, temperature: 0.0, stream: true, polish_temperature: 0.8 }
  variables:
    weather_condition: ["clear_sky","overcast"]
    lighting_condition: ["sunrise","night"]
    road_condition: ["dry","puddles"]

cosmos:
  model_version: "ct25"
  parameters:
    sigma: 90
    seed: null
    guidance: 3
    num_steps: 35
    inference_name: "cosmos_transfer_inference"
    modalities: ["edge"]
    weights: { edge: 1.0 }
    positive_prompt: "cinematic, photorealistic ..."
    negative_prompt: "cartoonish, low quality ..."
```

--- 

If you need additional fields documented, please point to your specific config(s) and we will extend this guide. 


