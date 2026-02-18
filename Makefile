# ============================================================
# 🎯 YOLO11 CCTV Emulation - Project Runner
# ============================================================
# Usage:
#   make setup      - Create venv & install dependencies
#   make label      - Open the image labeling GUI
#   make train      - Train the YOLO model
#   make test       - Run inference/testing
#   make cctv       - Run the CCTV emulation
#   make help       - Show all available commands
# ============================================================

PYTHON     := python3
VENV       := venv
VENV_BIN   := $(VENV)/bin
PIP        := $(VENV_BIN)/pip
PY         := $(VENV_BIN)/python

# Training defaults (override with: make train EPOCHS=200 BATCH=8)
EPOCHS     ?= 100
BATCH      ?= 16
PATIENCE   ?= 50
DATA       ?= simple_dataset/data.yaml
DEVICE     ?=

# Test defaults
CONF       ?= 0.25
MODEL      ?=

# ============================================================
# SETUP
# ============================================================

.PHONY: setup
activate:
	@echo "To activate the virtual environment, run:"
	@echo "source $(VENV)/bin/activate"

setup: $(VENV)/bin/activate ## Create venv and install all dependencies
	@echo "✅ Setup complete! Run 'make help' to see available commands."

$(VENV)/bin/activate:
	@echo "📦 Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "📥 Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Virtual environment ready."

.PHONY: install
install: $(VENV)/bin/activate ## Install/update dependencies from requirements.txt
	$(PIP) install -r requirements.txt

# ============================================================
# LABELING
# ============================================================

.PHONY: label
label: $(VENV)/bin/activate ## Open the image labeling GUI
	@echo "🏷️  Opening AI Image Labeling Tool..."
	$(PY) src/labeling_app.py

# ============================================================
# TRAINING
# ============================================================

.PHONY: train
train: $(VENV)/bin/activate ## Train the YOLO model (EPOCHS=100 BATCH=16 DATA=simple_dataset/data.yaml)
	@echo "🏋️  Starting YOLO training..."
	@echo "   Epochs:    $(EPOCHS)"
	@echo "   Batch:     $(BATCH)"
	@echo "   Patience:  $(PATIENCE)"
	@echo "   Data:      $(DATA)"
	$(PY) src/train.py \
		--data $(DATA) \
		--epochs $(EPOCHS) \
		--batch $(BATCH) \
		--patience $(PATIENCE) \
		$(if $(DEVICE),--device $(DEVICE),) \
		$(if $(MODEL),--model $(MODEL),)

# ============================================================
# TESTING
# ============================================================

.PHONY: test
test: $(VENV)/bin/activate ## Run inference on images (CONF=0.25 MODEL=auto)
	@echo "🔍 Running YOLO inference..."
	$(PY) src/test.py \
		--conf $(CONF) \
		$(if $(MODEL),--model $(MODEL),)

.PHONY: validate
validate: $(VENV)/bin/activate ## Run YOLO validation metrics
	@echo "📊 Running validation..."
	$(PY) src/test.py --validate \
		$(if $(MODEL),--model $(MODEL),)

# ============================================================
# CCTV EMULATION
# ============================================================



.PHONY: cctv
cctv: $(VENV)/bin/activate ## Run the 5-camera CCTV emulation
	@echo "📹 Starting CCTV Emulation (Default: Webcam)..."
	@echo "   Press 's' in the window to toggle between Webcam and Stream."
	@echo "   Ensure streaming server is running ('make stream') if switching to stream."
	$(PY) src/cctv_emulation.py

.PHONY: stream
stream: $(VENV)/bin/activate ## Run just the IP Camera Streaming Server
	@echo "� Starting Streaming Server at http://0.0.0.0:5001..."
	$(PY) src/stream_server.py


# ============================================================
# UTILITIES
# ============================================================

.PHONY: clean
clean: ## Remove training runs and cache files
	@echo "🗑️  Cleaning up..."
	rm -rf runs/
	find . -name "*.cache" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned."

.PHONY: clean-all
clean-all: clean ## Remove everything including venv
	@echo "🗑️  Removing virtual environment..."
	rm -rf $(VENV)
	@echo "✅ Full clean done. Run 'make setup' to start fresh."

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "🎯 YOLO11 CCTV Emulation - Available Commands"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "📝 Override defaults:"
	@echo "  make train EPOCHS=200 BATCH=8 DATA=labeled_dataset/data.yaml"
	@echo "  make test CONF=0.5 MODEL=path/to/model.pt"
	@echo "  make train DEVICE=mps"
	@echo ""

.DEFAULT_GOAL := help
