   # Experiment scripts (final results in the paper)

   Each script runs as one Google Colab cell and reads the Medical Segmentation Decathlon Task01 archive from Google Drive.

   - colab_v11_all.py: main experiment, 30 held-out volumes, all repair methods, Dice and statistics (GPU)
   - colab_v12_baselines.py: simple baselines and prediction of segmentation failure (CPU)
   - colab_v13_specificity.py: false alarms on noise, motion and bias field (CPU)
   - colab_v14_slice_gate.py: per-slice score for random spikes (CPU)
   - colab_figures.py: Figures 1 and 3 of the paper

   The U-Net checkpoint is trained by ../colab_v8_complete.py.
