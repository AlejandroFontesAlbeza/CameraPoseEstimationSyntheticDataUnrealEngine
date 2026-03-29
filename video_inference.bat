@echo off
echo Setting up inference parameters...can take a moment if using GPU for the first time.
python -m src.inference.main %*
