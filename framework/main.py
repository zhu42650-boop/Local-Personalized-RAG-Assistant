import os
import sys

from ui.window import launch_ui


def main():
    config_path = os.path.join(os.path.dirname(__file__), "config", "settings.yaml")
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    launch_ui(os.path.abspath(config_path))


if __name__ == "__main__":
    main()
