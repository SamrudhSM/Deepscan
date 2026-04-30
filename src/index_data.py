import faiss
import numpy as np

# Load the flat DNA for FAISS
dna_vectors = np.load("features_flat.npy").astype('float32')

# Initialize the index (HuBERT dimension = 768)
index = faiss.IndexFlatL2(768) 
index.add(dna_vectors)

# Save the index file
faiss.write_index(index, "forensic_voices.index")
print(f"✅ FAISS Index Built! {index.ntotal} voice signatures ready for search.")