# Metropolis SDG for Smart Cities 0.1.0 (18 Nov 2025)

## New Features

- End-to-end SDG workflow:
  - Stage 1: CARLA 0.9.16 ground-truth generation (RGB, Depth, Semantic/Instance Segmentation, Normals, Edges, masks, ODVG JSONs)
  - Stage 2: Cosmos-Transfer2.5 video augmentation with Gradio client
  - Stage 3: SoM-aligned post-processing and VLM-ready artifacts
- Docker Compose deployment:
  - Homogeneous mode: single-host stack for NIM services, Cosmos-Transfer, CARLA, and Jupyter Workbench
  - Heterogeneous mode: split NIM stack and Workbench across hosts with `NIM_HOST`
- One-command deploy script (`deploy/compose/deploy.sh`):
  - `./deploy.sh` (homogeneous), `./deploy.sh nim`, `./deploy.sh workbench`
  - Per-service GPU pinning via `*_GPU_ID` and `*_GPU_COUNT`
  - Health checks guidance and URLs printed
- Notebook (`notebooks/carla_synthetic_data_generation.ipynb`):
  - Self-guided flow: ground truth generation → captioning → template/prompt generation → augmentation → post-processing
  - NIM endpoints and ports derived from env
- Config assets and samples:
  - CARLA scenario JSON and camera YAML examples
  - Augmentation YAML (`modules/augmentation/configs/config_carla.yaml`)
  - Class filter YAML for masked edges

## Improvements

- README updates:
  - Quickstart with clear homogeneous/heterogeneous steps
  - Troubleshooting covers image access errors, health checks, GPU runtime setup, Docker permissions, port issues, and CARLA GPU usage.
- Advanced configuration guide (`data/docs/advanced_configuration.md`):
  - Field-by-field documentation for env, CARLA JSON/YAML, augmentation YAML
  - Validation rules and type constraints, recommendations, and examples
  - Notes on external reference configs (Inverted AI examples)
- Deploy UX:
  - `workbench` mode prompts to confirm `NIM_HOST` loaded from `deploy/compose/env`
  - README clarifies `NIM_HOST` in `env` overrides previously exported values

## Bug Fixes

- Robust config validation for augmentation pipeline (required sections and fields)
- Safer handling of missing seeds and prompt files
- Graceful errors for missing controls per selected modalities in Cosmos execution
