# /// script
# requires-python = "==3.11.*"
# dependencies = [
#   "tensorflow==2.15.0",
#   "tf2onnx",
#   "numpy==1.23.5",
# ]
# ///
import pathlib
import subprocess
import sys

def bulk_convert_tf2onnx():
    # 1. Define the subfolder for source files
    source_dir = pathlib.Path("./tflite")

    # 2. Define the output folder (current script directory)
    output_dir = pathlib.Path(".")

    # Find all .tflite files in the subfolder
    tflite_files = list(source_dir.glob("*.tflite"))

    if not tflite_files:
        print(f"No .tflite files found in {source_dir.absolute()}")
        return

    for tflite_file in tflite_files:
        # Save output in the current script folder instead of the subfolder
        onnx_file = output_dir / tflite_file.with_suffix(".onnx").name

        print(f"Converting {tflite_file.name} -> {onnx_file.name}...")

        cmd = [
            sys.executable,
            "-m", "tf2onnx.convert",
            "--tflite", str(tflite_file),
            "--output", str(onnx_file),
            "--opset", "13",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Successfully converted to {onnx_file.absolute()}")
        else:
            print(f"Error converting {tflite_file.name}:")
            print(result.stderr)

if __name__ == "__main__":
    bulk_convert_tf2onnx()