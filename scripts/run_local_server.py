import os


def export_config(override_env_with: str | None = None) -> None:
    env = override_env_with or os.getenv("ENV")
    with open(f"config/{env}.json", encoding="utf-8") as f:
        os.environ["CONFIG"] = f.read()


if __name__ == "__main__":
    export_config()
    os.system("pdm run local_server")
