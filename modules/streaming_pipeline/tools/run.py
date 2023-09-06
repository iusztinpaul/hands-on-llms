from beam import App, Image, Runtime, Volume, VolumeType

streaming_app = App(
    name="streaming_pipeline",
    runtime=Runtime(
        cpu=4,
        memory="32Gi",
        # TODO: Install requirements using Poetry & custom commands.
        image=Image(python_version="python3.10", python_packages="requirements.txt"),
    ),
    volumes=[
        Volume(path="./logs", name="logs", volume_type=VolumeType.Persistent),
        Volume(
            path="./model_cache", name="model_cache", volume_type=VolumeType.Persistent
        ),
    ],
)


@streaming_app.run()
def run(
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = None,
):
    from streaming_pipeline import initialize

    # Be sure to initialize the environment variables before importing any other modules.
    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)

    from streaming_pipeline.flow import build as flow_builder

    flow = flow_builder(model_cache_dir)

    from bytewax.testing import run_main

    run_main(flow)

    return flow
