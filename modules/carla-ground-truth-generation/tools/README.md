# Tools Documentation

## show_sensor.py

A utility for quickly previewing camera perspectives in CARLA log scenes. This tool helps verify sensor placement and configuration before running the full ground truth generation pipeline.

### Purpose

When configuring camera sensors for ground truth generation, it's important to verify that cameras are positioned correctly. `show_sensor.py` provides a fast way to:
- Preview camera viewpoints without generating full output data
- Test different sensor configurations quickly
- Verify camera placement in specific scenes from log files

### Usage

```bash
python tools/show_sensor.py -f <log_file> --sensors <sensor_config> [options]
```

### Command Line Options

| Parameter | Description | Example |
|-----------|-------------|----------|
| `-f, --file` | Path to CARLA log file | `example_data/wrong_way_scenarios_fixed/scenario_0/1753738530_carla_Town10HD_wrong_way_0_agent_0_fps30_output_log.log` |
| `--sensors` | Sensor configuration YAML file | `config/third_person_camera.yaml` |
| `-c, --camera` | Actor ID to track (optional) | `4641` |
| `-s, --start` | Start time in seconds | `0.0` |
| `-d, --duration` | Duration to preview | `5.0` |

### Example

Preview the third-person camera view from a wrong-way scenario:

```bash
python tools/show_sensor.py \
    -f example_data/wrong_way_scenarios_fixed/scenario_0/1753738530_carla_Town10HD_wrong_way_0_agent_0_fps30_output_log.log \
    --sensors config/third_person_camera.yaml \
    -s 0.0 -d 5.0
```

### Tips

1. Use this tool to quickly iterate on sensor configurations
2. Adjust camera positions in the YAML file and re-run to see changes immediately
3. Once satisfied with the camera placement, use `carla_cosmos_gen.py` to generate full ground truth data
