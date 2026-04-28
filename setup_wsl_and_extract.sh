#!/bin/bash
# setup_wsl_and_extract.sh
# Run this script INSIDE WSL after a system restart.
# It installs babeltrace, downloads the Apache dataset from Zenodo,
# and processes both Apache and ELK into the TraceReconstruction format.
#
# Usage (from WSL terminal):
#   bash /mnt/d/Naser/3/setup_wsl_and_extract.sh

set -e

WORKSPACE="/mnt/d/Naser/3"
DATASETS_DIR="$WORKSPACE/TraceReconstruction-main/Datasets"
ELK_TRACE_DIR="$WORKSPACE/elk-dataset-repo/Trace-RawData"
APACHE_WORK_DIR="$WORKSPACE/apache-raw"

echo "======================================================"
echo " Step 1: Install dependencies"
echo "======================================================"
sudo apt-get update -qq
sudo apt-get install -y babeltrace python3 python3-pip wget curl
pip3 install numpy babeltrace 2>/dev/null || true

# babeltrace Python module path
BT_PYPATH=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "/usr/local/lib/python3/dist-packages")
echo "babeltrace installed. Python site: $BT_PYPATH"

echo "======================================================"
echo " Step 2: Reassemble and extract ELK raw traces"
echo "======================================================"
ELK_EXTRACT_DIR="$WORKSPACE/elk-extracted"
mkdir -p "$ELK_EXTRACT_DIR"

echo "  Concatenating split files..."
cat "$ELK_TRACE_DIR"/elktrace_* > "$ELK_EXTRACT_DIR/elktrace.tar.gz"
echo "  Extracting tar.gz..."
tar -xzf "$ELK_EXTRACT_DIR/elktrace.tar.gz" -C "$ELK_EXTRACT_DIR"
echo "  Extraction complete."
ls "$ELK_EXTRACT_DIR"

echo "======================================================"
echo " Step 3: Extract ELK syscall sequences"
echo "======================================================"
python3 << 'PYEOF'
import babeltrace
import os
import sys

ELK_EXTRACT_DIR = os.environ.get("ELK_EXTRACT_DIR", "/mnt/d/Naser/3/elk-extracted")
OUT_FILE = "/mnt/d/Naser/3/TraceReconstruction-main/Datasets/elk/elk_syscalls.txt"

# Find trace directories (LTTng CTF subdirectories)
trace_dirs = []
for root, dirs, files in os.walk(ELK_EXTRACT_DIR):
    if "metadata" in files:
        trace_dirs.append(root)

print(f"Found {len(trace_dirs)} trace directories")

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
event_count = 0
with open(OUT_FILE, "w") as fout:
    for trace_dir in sorted(trace_dirs):
        try:
            tc = babeltrace.TraceCollection()
            tc.add_trace(trace_dir, "ctf")
            for event in tc.events:
                if "syscall" in event.name:
                    name = event.name.replace("syscall_entry_", "").replace("syscall_exit_", "").replace("entry_", "").replace("exit_", "")
                    fout.write(name + "\n")
                    event_count += 1
                    if event_count % 1000000 == 0:
                        print(f"  {event_count:,} events written...", flush=True)
        except Exception as e:
            print(f"  Warning: {trace_dir}: {e}", file=sys.stderr)

print(f"Total ELK events written: {event_count:,}")
PYEOF

echo "======================================================"
echo " Step 4: Download Apache dataset from Zenodo"
echo "======================================================"
mkdir -p "$APACHE_WORK_DIR"
cd "$APACHE_WORK_DIR"

if [ ! -f "data.zip" ]; then
    echo "  Downloading data.zip from Zenodo (~1.1GB)..."
    wget -q --show-progress -O data.zip "https://zenodo.org/records/4091287/files/data.zip"
else
    echo "  data.zip already exists, skipping download."
fi

echo "  Unzipping..."
unzip -q -o data.zip -d apache_data

echo "======================================================"
echo " Step 5: Extract Apache syscall sequences"
echo "======================================================"

python3 << 'PYEOF'
import babeltrace
import os
import sys

APACHE_DATA = "/mnt/d/Naser/3/apache-raw/apache_data/data/requests"
TRAIN_OUT   = "/mnt/d/Naser/3/TraceReconstruction-main/Datasets/apache/train_syscalls.txt"
TEST_OUT    = "/mnt/d/Naser/3/TraceReconstruction-main/Datasets/apache/test_syscalls.txt"

os.makedirs(os.path.dirname(TRAIN_OUT), exist_ok=True)

def extract_syscalls(trace_path):
    """Extract syscall names from an LTTng kernel trace directory."""
    syscalls = []
    tc = babeltrace.TraceCollection()
    tc.add_trace(trace_path, "ctf")
    for event in tc.events:
        if "syscall" in event.name:
            name = (event.name
                    .replace("syscall_entry_", "")
                    .replace("syscall_exit_", "")
                    .replace("entry_", "")
                    .replace("exit_", ""))
            syscalls.append(name)
    return syscalls

print("Extracting train trace...")
train_syscalls = extract_syscalls(os.path.join(APACHE_DATA, "train", "kernel"))
print(f"  Train events: {len(train_syscalls):,}")
with open(TRAIN_OUT, "w") as f:
    f.write("\n".join(train_syscalls))

print("Extracting test trace...")
test_syscalls = extract_syscalls(os.path.join(APACHE_DATA, "test", "kernel"))
print(f"  Test events: {len(test_syscalls):,}")
with open(TEST_OUT, "w") as f:
    f.write("\n".join(test_syscalls))

print("Done. Now run preprocess_apache.py from Windows.")
PYEOF

echo "======================================================"
echo " Step 6: Run Python preprocessing scripts (on Windows)"
echo "======================================================"
echo ""
echo "  ELK:    The elk_syscalls.txt file is now ready."
echo "          Run on Windows: python TraceReconstruction-main/Datasets/elk/preprocess_elk.py"
echo ""
echo "  Apache: The train_syscalls.txt and test_syscalls.txt files are ready."
echo "          Run on Windows: python TraceReconstruction-main/Datasets/apache/preprocess_apache.py"
echo ""
echo "All done!"
