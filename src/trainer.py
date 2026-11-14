import os
import numpy as np
import tensorflow as tf
from src.model import build_unet

def train_and_save_model():
    os.makedirs('knowledge_base', exist_ok=True)
    model = build_unet((256, 256, 1))

    # Create dummy training data to initialize knowledge base weights
    X_dummy = np.random.rand(16, 256, 256, 1).astype(np.float32)
    y_dummy = np.random.randint(0, 2, size=(16, 256, 256, 1)).astype(np.float32)

    print("Training dummy/initial model for Knowledge Base...")
    model.fit(X_dummy, y_dummy, epochs=2, batch_size=4, verbose=1)

    model_path = os.path.join('knowledge_base', 'lake_unet_model.h5')
    model.save(model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == '__main__':
    train_and_save_model()
