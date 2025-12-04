import argparse
import time
from pathlib import Path
import mujoco

SIMULATE = False

parser = argparse.ArgumentParser(description="Launch MuJoCo passive viewer for the hand model.")
parser.add_argument(
    "--simulate",
    type=int,
    choices=[0, 1],
    default=int(SIMULATE),
    help="Set to 1 to step simulation, 0 to keep static view.",
)
parser.add_argument(
    "--xml-path",
    type=str,
    default=str(Path(__file__).resolve().parent.parent / "xml" / "myohuman.xml"),
    help="Path to the XML model; default is resolved relative to this script.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    simulate = bool(args.simulate)
    xml_path = Path(args.xml_path).expanduser().resolve()

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 90
        while viewer.is_running():
            if simulate:
                mujoco.mj_step(model, data)
                viewer.sync()