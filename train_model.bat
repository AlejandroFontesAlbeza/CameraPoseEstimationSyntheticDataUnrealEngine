@echo off
echo Setting up training parameters...can take a moment if using GPU for the first time.
python -m src.training.main %*
