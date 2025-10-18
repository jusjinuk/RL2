# PDDL Problem Generation Evaluation

Evaluate planning models on PDDL problem generation (Planetarium benchmark).

## Quick Start

```bash
bash pddl.sh
```

## Configuration

Edit `pddl.sh` to customize:

```bash
python pddl_eval.py \
  --name "Qwen/Qwen3-4B-Thinking-2507" \  # Model name
  --output-dir ./pddl/qwen3 \             # Output directory
  --completion-limit 4 \                   # Completions per problem
  --batch-size 64 \                        # Batch size
  --first-turn-max-new-tokens 1024 \      # First turn tokens
  --second-turn-max-new-tokens 1024 \     # Second turn tokens
  --evaluate \                             # Run PDDL validation
  --notes "pddl qwen3" \                  # CSV notes
  --passk-path results.csv \              # Pass@k CSV file
  --dataset-split train \                  # Dataset split
  --dataset-index 1 2 3 4                 # Problem subset (optional)
```

## Arguments

- `--name`: HuggingFace model path
- `--output-dir`: Results directory
- `--dataset`: Dataset name (default: BatsResearch/planetarium)
- `--dataset-split`: Split to use (train/test, default: test)
- `--completion-limit`: Completions per problem (default: 200)
- `--batch-size`: Generation batch size (default: 16)
- `--first-turn-max-new-tokens`: Max tokens for reasoning (default: 1024)
- `--second-turn-max-new-tokens`: Max tokens for completion (default: 1024)
- `--evaluate`: Enable PDDL equivalence checking
- `--passk-path`: CSV file for pass@k results
- `--notes`: Additional CSV metadata
- `--domain-path`: PDDL domains directory (default: ~/RL2/pddl)
- `--dataset-index`: Filter specific problems (e.g., `1 2 3 4`)

## Pipeline Stages

1. **Get Ready** (1_get_ready): Load dataset, domains, and model
2. **Generate** (2_generate): Batch PDDL generation
3. **Evaluate** (3_evaluate): Multiprocessing equivalence checking
4. **Wrapup** (4_wrapup): Calculate pass@k metrics

## Output Files

- `{output_dir}/*.json.gz`: Per-problem results
- `{output_dir}/timing_pddl_*.json`: Timing breakdown
- `results.csv`: Pass@k metrics across runs

## Evaluation Metrics

- **Parseable**: Valid PDDL syntax
- **Valid**: Semantically correct PDDL
- **Equivalent**: Functionally equivalent to ground truth

## Timing Analysis

```bash
# View timing for single run
cat pddl/qwen3/timing_pddl_*.json
```

## Installation

You must install the following tools to run PDDL evaluation.

### Prerequisites

Required system packages (if not already installed):
- `cmake`, `make`, `g++` for building VAL
- `flex`, `bison` (optional, for parser regeneration)

### 1. Install Go if uninstalled (for Apptainer)

```bash
# Set Go version and platform
export GOVERSION=1.23.6 OS=linux ARCH=amd64

# Download and install Go to home directory (no sudo required)
wget -O /tmp/go${GOVERSION}.${OS}-${ARCH}.tar.gz \
  https://dl.google.com/go/go${GOVERSION}.${OS}-${ARCH}.tar.gz

# Extract to home directory
tar -C $HOME -xzf /tmp/go${GOVERSION}.${OS}-${ARCH}.tar.gz

# Add to PATH
echo 'export PATH=$HOME/go/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
go version
```

### 2. Install Apptainer

```bash
# Clone Apptainer repository
git clone https://github.com/apptainer/apptainer.git
cd apptainer
git checkout v1.4.3

# Configure with local prefix (no sudo required)
./mconfig --prefix=$HOME/.local

# Build and install
cd builddir
make
make install

# Verify installation
apptainer version
```

### 3. Install Fast-Downward Planner

Fast-Downward runs inside an Apptainer container. You need to pull the container image and create a wrapper script.

```bash
# Pull Fast-Downward container image
cd ~
apptainer pull fast-downward.sif docker://aibasel/downward:latest

# Create wrapper script for easy access
cat > ~/.local/bin/fast-downward << 'EOF'
#!/bin/bash
# Fast-Downward wrapper script
exec apptainer run ~/fast-downward.sif "$@"
EOF
chmod +x ~/.local/bin/fast-downward

# Test installation
fast-downward --help
```

**Note**: Fast-Downward always runs through Apptainer, so you'll see container messages when executing it.

### 4. Install VAL Plan Validator

```bash
# Clone VAL repository
cd ~
git clone https://github.com/KCL-Planning/VAL.git

# Build VAL
cd VAL
mkdir -p build/linux64
cd build/linux64
cmake ../..
make -j4

# Copy binaries to local bin directory
cp bin/* ~/.local/bin/

# Verify installation
Validate --help
```

### 5. Set up PATH

Ensure `~/.local/bin` is in your PATH. 
If not:

```bash
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Verification

```bash
# Check all tools are accessible
go version                    # Should show Go 1.23.6+
apptainer version             # Should show 1.4.3
fast-downward --help          # Should show Fast-Downward help
Validate --help               # Should show VAL validator help
```

